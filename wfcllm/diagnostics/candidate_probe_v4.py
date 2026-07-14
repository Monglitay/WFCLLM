"""Pure helpers for pre-preregistration V4 mechanism candidate probes."""

from __future__ import annotations

import ast
import copy
import hashlib
import hmac
import json
import math
import re
import stat
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PROBE_SCHEMA_VERSION = "wfcllm-v4-candidate-probe/v1"
STRUCTURAL_SCHEMA_VERSION = "wfcllm-v4-structural-context/v1"
_HMAC_HEADER = b"WFCLLM_BATCH_INVARIANT_SEMANTIC_V4_PROBE\0"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_BUILTIN_NAMES = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "int",
        "len",
        "list",
        "map",
        "max",
        "min",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
)


class ProbeSecret:
    """Opaque V4 diagnostic secret used only by pre-preregistration probes."""

    __slots__ = ("_material",)

    def __init__(self, material: bytes) -> None:
        if not isinstance(material, bytes) or len(material) < 32:
            raise ValueError("probe secret must contain at least 32 bytes")
        self._material = bytes(material)

    @classmethod
    def from_material_for_test(cls, material: bytes) -> ProbeSecret:
        return cls(material)

    def __repr__(self) -> str:
        return "ProbeSecret(<redacted>)"


def load_probe_secret(path: str | Path) -> ProbeSecret:
    key_path = Path(path)
    try:
        file_stat = key_path.stat()
    except OSError as exc:
        raise ValueError(f"failed to stat probe key file: {key_path}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError("probe key path must be a regular file")
    if stat.S_IMODE(file_stat.st_mode) != 0o600:
        raise ValueError("probe key file mode must be exactly 0600")
    try:
        return ProbeSecret(key_path.read_bytes())
    except OSError as exc:
        raise ValueError(f"failed to read probe key file: {key_path}") from exc


def _derive_bits(
    secret: ProbeSecret,
    *,
    domain: bytes,
    message: bytes,
    bit_count: int,
) -> tuple[int, ...]:
    if not isinstance(secret, ProbeSecret):
        raise ValueError("secret must be ProbeSecret")
    if not isinstance(message, bytes):
        raise ValueError("message must be bytes")
    if isinstance(bit_count, bool) or not isinstance(bit_count, int) or bit_count <= 0:
        raise ValueError("bit_count must be a positive integer")
    output = bytearray()
    counter = 0
    byte_count = (bit_count + 7) // 8
    while len(output) < byte_count:
        counter += 1
        output.extend(
            hmac.new(
                secret._material,
                _HMAC_HEADER + domain + b"\0" + message + counter.to_bytes(4, "big"),
                hashlib.sha256,
            ).digest()
        )
    bits: list[int] = []
    for byte in output[:byte_count]:
        bits.extend((byte >> shift) & 1 for shift in range(8))
    return tuple(bits[:bit_count])


def derive_projection_bits(
    secret: ProbeSecret,
    message: bytes,
    *,
    bit_count: int,
) -> tuple[int, ...]:
    return _derive_bits(
        secret,
        domain=b"v4-probe/projection/signature",
        message=message,
        bit_count=bit_count,
    )


def derive_target_bits(
    secret: ProbeSecret,
    message: bytes,
    *,
    bit_count: int,
) -> tuple[int, ...]:
    return _derive_bits(
        secret,
        domain=b"v4-probe/target/unit",
        message=message,
        bit_count=bit_count,
    )


class _ScopedIdentifierNormalizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.mapping: dict[str, str] = {}

    def _slot(self, name: str) -> str:
        if name in _BUILTIN_NAMES:
            return f"builtin_{name}"
        if name not in self.mapping:
            self.mapping[name] = f"identifier_{len(self.mapping)}"
        return self.mapping[name]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id=self._slot(node.id), ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        return ast.copy_location(
            ast.arg(
                arg=self._slot(node.arg),
                annotation=(
                    self.visit(node.annotation) if node.annotation is not None else None
                ),
                type_comment=node.type_comment,
            ),
            node,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        normalized = self.generic_visit(node)
        assert isinstance(normalized, ast.FunctionDef)
        normalized.name = self._slot(node.name)
        return normalized

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        normalized = self.generic_visit(node)
        assert isinstance(normalized, ast.AsyncFunctionDef)
        normalized.name = self._slot(node.name)
        return normalized


def _parse_fragment(text: str, *, field: str) -> ast.Module | None:
    if text == "<BOS>":
        return None
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{field} must be non-empty source or <BOS>")
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ValueError(f"failed to parse {field} context") from exc


def _dataflow_edges(
    previous: ast.Module | None,
    current: ast.Module,
) -> tuple[str, ...]:
    last_store: dict[str, str] = {}
    edges: list[str] = []
    for segment, tree in (("previous", previous), ("current", current)):
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Name) or node.id.startswith("builtin_"):
                continue
            if isinstance(node.ctx, ast.Store):
                last_store[node.id] = segment
            elif isinstance(node.ctx, ast.Load) and node.id in last_store:
                edges.append(f"{node.id}:{last_store[node.id]}->{segment}")
    return tuple(sorted(edges))


@dataclass(frozen=True)
class CanonicalStructuralUnit:
    role: str
    representation: str
    representation_bytes: tuple[int, ...]
    representation_sha256: str
    context_sha256: str
    unit_id: str


def canonical_structural_unit(
    *,
    role: str,
    previous: str,
    current: str,
) -> CanonicalStructuralUnit:
    """Build identifier-normalized AST and bounded local data-flow evidence."""

    if not isinstance(role, str) or not role or "\0" in role:
        raise ValueError("role must be a non-empty NUL-free string")
    previous_tree = _parse_fragment(previous, field="previous")
    current_tree = _parse_fragment(current, field="current")
    assert current_tree is not None
    normalizer = _ScopedIdentifierNormalizer()
    normalized_previous = (
        normalizer.visit(previous_tree) if previous_tree is not None else None
    )
    normalized_current = normalizer.visit(current_tree)
    assert isinstance(normalized_current, ast.Module)
    if normalized_previous is not None:
        assert isinstance(normalized_previous, ast.Module)
    ast.fix_missing_locations(normalized_current)
    if normalized_previous is not None:
        ast.fix_missing_locations(normalized_previous)
    payload = {
        "schema_version": STRUCTURAL_SCHEMA_VERSION,
        "role": role,
        "previous_ast": (
            "<BOS>"
            if normalized_previous is None
            else ast.dump(normalized_previous, annotate_fields=True, include_attributes=False)
        ),
        "current_ast": ast.dump(
            normalized_current,
            annotate_fields=True,
            include_attributes=False,
        ),
        "bounded_dataflow": _dataflow_edges(
            normalized_previous,
            normalized_current,
        ),
    }
    representation = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    representation_sha256 = hashlib.sha256(representation.encode("utf-8")).hexdigest()
    representation_bytes = tuple(bytes.fromhex(representation_sha256))
    unit_id = hashlib.sha256(
        (
            STRUCTURAL_SCHEMA_VERSION
            + "\0"
            + role
            + "\0"
            + representation_sha256
        ).encode("utf-8")
    ).hexdigest()
    return CanonicalStructuralUnit(
        role=role,
        representation=representation,
        representation_bytes=representation_bytes,
        representation_sha256=representation_sha256,
        context_sha256=representation_sha256,
        unit_id=unit_id,
    )


@dataclass(frozen=True)
class StructuralEvidence:
    unit_id: str
    context_sha256: str
    representation: tuple[int, ...]
    quantized_values: tuple[int, ...]
    erasure_mask: tuple[bool, ...]
    signature_bits: tuple[int, ...]
    target_bits: tuple[int, ...]
    matches: int
    numerator: int
    denominator: int

    def with_erasure_mask(self, mask: Sequence[bool]) -> StructuralEvidence:
        return replace(self, erasure_mask=tuple(mask))

    def with_representation(self, values: Sequence[int]) -> StructuralEvidence:
        normalized = tuple(int(value) for value in values)
        return replace(
            self,
            representation=normalized,
            quantized_values=normalized,
        )


@dataclass(frozen=True)
class StructuralExtraction:
    parse_ok: bool
    units: tuple[CanonicalStructuralUnit, ...]
    erasure_counts: dict[str, int]


def _statement_shell_source(statement: ast.stmt) -> str:
    shell = copy.deepcopy(statement)
    for field_name in ("body", "orelse", "finalbody"):
        value = getattr(shell, field_name, None)
        if isinstance(value, list):
            if value:
                setattr(shell, field_name, [ast.Pass()])
            else:
                setattr(shell, field_name, [])
    handlers = getattr(shell, "handlers", None)
    if isinstance(handlers, list):
        for handler in handlers:
            if isinstance(handler, ast.ExceptHandler):
                handler.body = [ast.Pass()]
    cases = getattr(shell, "cases", None)
    if isinstance(cases, list):
        for case in cases:
            if isinstance(case, ast.match_case):
                case.body = [ast.Pass()]
    ast.fix_missing_locations(shell)
    return ast.unparse(shell).strip().replace("\r\n", "\n")


def _nested_statement_groups(
    statement: ast.stmt,
) -> tuple[tuple[str, str, tuple[ast.stmt, ...]], ...]:
    parent_type = type(statement).__name__
    groups: list[tuple[str, str, tuple[ast.stmt, ...]]] = []
    for field_name in ("body", "orelse", "finalbody"):
        value = getattr(statement, field_name, None)
        if isinstance(value, list) and value and all(
            isinstance(item, ast.stmt) for item in value
        ):
            groups.append((parent_type, field_name, tuple(value)))
    handlers = getattr(statement, "handlers", None)
    if isinstance(handlers, list):
        for handler in handlers:
            if isinstance(handler, ast.ExceptHandler) and handler.body:
                groups.append(("ExceptHandler", "body", tuple(handler.body)))
    cases = getattr(statement, "cases", None)
    if isinstance(cases, list):
        for case in cases:
            if isinstance(case, ast.match_case) and case.body:
                groups.append(("match_case", "body", tuple(case.body)))
    return tuple(groups)


def extract_structural_units(
    code: str,
    *,
    max_unit_bytes: int = 4096,
    max_context_bytes: int = 8192,
) -> StructuralExtraction:
    if not isinstance(code, str):
        return StructuralExtraction(
            parse_ok=False,
            units=(),
            erasure_counts={"parse_failure": 1},
        )
    for value, name in (
        (max_unit_bytes, "max_unit_bytes"),
        (max_context_bytes, "max_context_bytes"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return StructuralExtraction(
            parse_ok=False,
            units=(),
            erasure_counts={"parse_failure": 1},
        )
    candidates: list[CanonicalStructuralUnit] = []
    erasures: defaultdict[str, int] = defaultdict(int)
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    def visit_group(
        statements: Sequence[ast.stmt],
        *,
        parent_type: str,
        field_name: str,
    ) -> None:
        previous = "<BOS>"
        for statement in statements:
            current = _statement_shell_source(statement)
            if len(current.encode("utf-8")) > max_unit_bytes:
                erasures["unit_too_large"] += 1
                previous = current
                continue
            role = f"{type(statement).__name__}|{parent_type}|{field_name}"
            serialized_size = len(
                (role + "\0" + previous + "\0" + current).encode("utf-8")
            )
            if serialized_size > max_context_bytes:
                erasures["context_too_large"] += 1
                previous = current
                continue
            candidates.append(
                canonical_structural_unit(
                    role=role,
                    previous=previous,
                    current=current,
                )
            )
            previous = current
            for child_parent, child_field, child_statements in _nested_statement_groups(
                statement
            ):
                visit_group(
                    child_statements,
                    parent_type=child_parent,
                    field_name=child_field,
                )

    for function in functions:
        visit_group(
            function.body,
            parent_type=type(function).__name__,
            field_name="body",
        )
    counts: defaultdict[str, int] = defaultdict(int)
    for unit in candidates:
        counts[unit.unit_id] += 1
    duplicate_ids = {unit_id for unit_id, count in counts.items() if count > 1}
    units = tuple(unit for unit in candidates if unit.unit_id not in duplicate_ids)
    if duplicate_ids:
        erasures["duplicate_unit_id"] += sum(
            unit.unit_id in duplicate_ids for unit in candidates
        )
    return StructuralExtraction(
        parse_ok=True,
        units=units,
        erasure_counts=dict(sorted(erasures.items())),
    )


def score_structural_unit(
    unit: CanonicalStructuralUnit,
    secret: ProbeSecret,
    *,
    bit_count: int,
) -> StructuralEvidence:
    signature = derive_projection_bits(
        secret,
        bytes(unit.representation_bytes),
        bit_count=bit_count,
    )
    target = derive_target_bits(
        secret,
        unit.unit_id.encode("ascii"),
        bit_count=bit_count,
    )
    matches = sum(left == right for left, right in zip(signature, target, strict=True))
    return StructuralEvidence(
        unit_id=unit.unit_id,
        context_sha256=unit.context_sha256,
        representation=unit.representation_bytes,
        quantized_values=unit.representation_bytes,
        erasure_mask=(False,) * bit_count,
        signature_bits=signature,
        target_bits=target,
        matches=matches,
        numerator=2 * matches - bit_count,
        denominator=bit_count,
    )


_EVIDENCE_FIELDS = (
    "unit_id",
    "context_sha256",
    "representation",
    "quantized_values",
    "erasure_mask",
    "signature_bits",
    "target_bits",
    "matches",
    "numerator",
    "denominator",
)


def exact_evidence_mismatches(
    reference: StructuralEvidence,
    candidate: StructuralEvidence,
) -> tuple[str, ...]:
    return tuple(
        field
        for field in _EVIDENCE_FIELDS
        if getattr(reference, field) != getattr(candidate, field)
    )


@dataclass(frozen=True)
class StructuralCodeScore:
    evidence: tuple[StructuralEvidence, ...]
    numerator: int
    denominator: int
    independent_units: int
    eligible: bool
    erasure_counts: dict[str, int]

    @property
    def score(self) -> float:
        return self.numerator / self.denominator if self.denominator else 0.0

    def exact_mismatches(self, candidate: StructuralCodeScore) -> tuple[str, ...]:
        fields = (
            "evidence",
            "numerator",
            "denominator",
            "independent_units",
            "eligible",
            "erasure_counts",
        )
        return tuple(
            field
            for field in fields
            if getattr(self, field) != getattr(candidate, field)
        )


def score_structural_code(
    final_code: str,
    secret: ProbeSecret,
    *,
    bit_count: int,
    minimum_independent_units: int,
) -> StructuralCodeScore:
    if (
        isinstance(minimum_independent_units, bool)
        or not isinstance(minimum_independent_units, int)
        or minimum_independent_units <= 0
    ):
        raise ValueError("minimum_independent_units must be a positive integer")
    extraction = extract_structural_units(final_code)
    evidence = tuple(
        score_structural_unit(unit, secret, bit_count=bit_count)
        for unit in extraction.units
    )
    numerator = sum(item.numerator for item in evidence)
    denominator = sum(item.denominator for item in evidence)
    return StructuralCodeScore(
        evidence=evidence,
        numerator=numerator,
        denominator=denominator,
        independent_units=len(evidence),
        eligible=len(evidence) >= minimum_independent_units,
        erasure_counts=extraction.erasure_counts,
    )


@dataclass(frozen=True)
class MarginReplay:
    reference_erasure_mask: tuple[bool, ...]
    candidate_erasure_mask: tuple[bool, ...]
    erasure_mask_mismatch_count: int
    signature_mismatch_count: int
    eligible_bit_count: int
    exact: bool


def margin_replay(
    *,
    reference_dots: Sequence[int],
    candidate_dots: Sequence[int],
    absolute_dot_bound: int,
) -> MarginReplay:
    if len(reference_dots) != len(candidate_dots) or not reference_dots:
        raise ValueError("projection dot vectors must have the same positive length")
    if (
        isinstance(absolute_dot_bound, bool)
        or not isinstance(absolute_dot_bound, int)
        or absolute_dot_bound < 0
    ):
        raise ValueError("absolute_dot_bound must be a non-negative integer")
    reference_mask = tuple(abs(int(value)) <= absolute_dot_bound for value in reference_dots)
    candidate_mask = tuple(abs(int(value)) <= absolute_dot_bound for value in candidate_dots)
    mask_mismatches = sum(
        left != right for left, right in zip(reference_mask, candidate_mask, strict=True)
    )
    signature_mismatches = 0
    eligible = 0
    for ref_dot, cand_dot, ref_erased, cand_erased in zip(
        reference_dots,
        candidate_dots,
        reference_mask,
        candidate_mask,
        strict=True,
    ):
        if ref_erased or cand_erased:
            continue
        eligible += 1
        signature_mismatches += int((int(ref_dot) >= 0) != (int(cand_dot) >= 0))
    return MarginReplay(
        reference_erasure_mask=reference_mask,
        candidate_erasure_mask=candidate_mask,
        erasure_mask_mismatch_count=mask_mismatches,
        signature_mismatch_count=signature_mismatches,
        eligible_bit_count=eligible,
        exact=mask_mismatches == 0 and signature_mismatches == 0,
    )


@dataclass(frozen=True)
class CandidateRecord:
    task_id: str
    attempt_index: int
    final_code: str
    final_code_sha256: str


@dataclass(frozen=True)
class ProbeContext:
    case_id: str
    category: str
    context_sha256: str
    role: str
    serialized: str
    token_count: int
    previous: str
    current: str


def parse_serialized_context(serialized: str) -> tuple[str, str, str]:
    if not isinstance(serialized, str):
        raise ValueError("serialized context must be a string")
    header = "WFCLLM_DYNAMIC_SEMANTIC_CONTEXT_V3\nrole="
    if not serialized.startswith(header):
        raise ValueError("serialized context has an invalid header")
    body = serialized[len(header) :]
    try:
        role, remainder = body.split("\nprevious=", 1)
        previous, current = remainder.split("\ncurrent=", 1)
    except ValueError as exc:
        raise ValueError("serialized context is missing a required field") from exc
    if not role or not previous or not current:
        raise ValueError("serialized context fields must be non-empty")
    return role, previous, current


def load_probe_contexts(path: str | Path) -> tuple[ProbeContext, ...]:
    context_path = Path(path)
    contexts: list[ProbeContext] = []
    seen_case_ids: set[str] = set()
    seen_hashes: set[str] = set()
    try:
        lines = context_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"failed to read context artifact: {context_path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid context JSON at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"context line {line_number} must be a JSON object")
        required = {
            "case_id",
            "category",
            "context_sha256",
            "role",
            "serialized",
            "token_count",
        }
        if not required.issubset(row):
            raise ValueError(f"context line {line_number} is missing required fields")
        serialized = row["serialized"]
        role, previous, current = parse_serialized_context(serialized)
        context_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        if context_sha256 != row["context_sha256"]:
            raise ValueError(f"context SHA-256 mismatch at line {line_number}")
        if role != row["role"]:
            raise ValueError(f"context role mismatch at line {line_number}")
        case_id = row["case_id"]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"context case ID at line {line_number} must be non-empty")
        if case_id in seen_case_ids or context_sha256 in seen_hashes:
            raise ValueError("probe contexts must have unique case IDs and hashes")
        category = row["category"]
        token_count = row["token_count"]
        if not isinstance(category, str) or not category:
            raise ValueError(f"context category at line {line_number} must be non-empty")
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
            raise ValueError(f"context token count at line {line_number} must be positive")
        contexts.append(
            ProbeContext(
                case_id=case_id,
                category=category,
                context_sha256=context_sha256,
                role=role,
                serialized=serialized,
                token_count=token_count,
                previous=previous,
                current=current,
            )
        )
        seen_case_ids.add(case_id)
        seen_hashes.add(context_sha256)
    if not contexts:
        raise ValueError("context artifact must contain at least one context")
    return tuple(contexts)


def load_candidate_ledger(
    path: str | Path,
    *,
    retry: int,
    allowed_task_ids: Sequence[str],
) -> tuple[CandidateRecord, ...]:
    ledger_path = Path(path)
    allowed = frozenset(allowed_task_ids)
    if not allowed:
        raise ValueError("allowed task IDs must not be empty")
    records: list[CandidateRecord] = []
    try:
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"failed to read candidate ledger: {ledger_path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid candidate JSON at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"candidate line {line_number} must be a JSON object")
        task_id = row.get("id")
        if task_id not in allowed:
            continue
        try:
            records.append(
                CandidateRecord(
                    task_id=task_id,
                    attempt_index=row["attempt_index"],
                    final_code=row["final_code"],
                    final_code_sha256=row["final_code_sha256"],
                )
            )
        except KeyError as exc:
            raise ValueError(f"candidate line {line_number} is missing required fields") from exc
    present = frozenset(record.task_id for record in records)
    if present != allowed:
        missing = sorted(allowed - present)
        raise ValueError(f"candidate ledger is missing allowed task IDs: {missing}")
    return validate_candidate_pool(records, retry=retry)


@dataclass(frozen=True)
class MarginProbeSummary:
    row_count: int
    bit_count: int
    erased_reference_bits: int
    erased_candidate_bits: int
    erasure_mask_mismatch_rows: int
    signature_mismatch_rows: int
    exact_rows: int


@dataclass(frozen=True)
class ShapeIsolatedProbeSummary:
    context_count: int
    schedule_names: tuple[str, ...]
    total_replays: int
    exact_replays: int
    exact_replay_rate: float
    cache_hit_miss_exact: bool
    physical_encode_calls: int
    mean_cache_miss_seconds_per_context: float
    mean_cache_hit_seconds_per_context: float


def probe_shape_isolated(
    contexts: Sequence[str],
    *,
    encode_one: Callable[[str], Sequence[int]],
) -> ShapeIsolatedProbeSummary:
    normalized = tuple(contexts)
    if not normalized or any(
        not isinstance(context, str) or not context for context in normalized
    ):
        raise ValueError("shape-isolated probe requires non-empty string contexts")
    if len(set(normalized)) != len(normalized):
        raise ValueError("shape-isolated probe contexts must be unique")

    def encode(context: str) -> tuple[int, ...]:
        values = tuple(int(value) for value in encode_one(context))
        if not values:
            raise ValueError("shape-isolated encoder must return discrete evidence")
        return values

    cache: dict[str, tuple[int, ...]] = {}
    miss_seconds = 0.0
    physical_calls = 0
    for context in normalized:
        started = time.perf_counter()
        cache[context] = encode(context)
        miss_seconds += time.perf_counter() - started
        physical_calls += 1

    schedules = (
        ("forward", normalized),
        ("reverse", tuple(reversed(normalized))),
        (
            "permutation",
            tuple(
                sorted(
                    normalized,
                    key=lambda value: hashlib.sha256(
                        ("v4-probe-permutation\0" + value).encode("utf-8")
                    ).digest(),
                )
            ),
        ),
    )
    exact_replays = 0
    total_replays = 0
    for _, schedule in schedules:
        for context in schedule:
            started = time.perf_counter()
            candidate = encode(context)
            miss_seconds += time.perf_counter() - started
            physical_calls += 1
            total_replays += 1
            exact_replays += candidate == cache[context]

    hit_seconds = 0.0
    cache_exact = True
    for context in normalized:
        started = time.perf_counter()
        cached = cache[context]
        hit_seconds += time.perf_counter() - started
        cache_exact = cache_exact and cached == cache[context]
    miss_count = physical_calls
    return ShapeIsolatedProbeSummary(
        context_count=len(normalized),
        schedule_names=tuple(name for name, _ in schedules),
        total_replays=total_replays,
        exact_replays=exact_replays,
        exact_replay_rate=exact_replays / total_replays,
        cache_hit_miss_exact=cache_exact and exact_replays == total_replays,
        physical_encode_calls=physical_calls,
        mean_cache_miss_seconds_per_context=miss_seconds / miss_count,
        mean_cache_hit_seconds_per_context=hit_seconds / len(normalized),
    )


def summarize_margin_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    absolute_dot_bound: int,
) -> MarginProbeSummary:
    row_count = 0
    bit_count = 0
    erased_reference = 0
    erased_candidate = 0
    mask_mismatch_rows = 0
    signature_mismatch_rows = 0
    exact_rows = 0
    for row in rows:
        try:
            discrete = row["discrete"]
            reference_dots = discrete["reference_projection_dots"]
            candidate_dots = discrete["candidate_projection_dots"]
        except (KeyError, TypeError) as exc:
            raise ValueError("margin row is missing raw projection dots") from exc
        replay = margin_replay(
            reference_dots=reference_dots,
            candidate_dots=candidate_dots,
            absolute_dot_bound=absolute_dot_bound,
        )
        row_count += 1
        bit_count += len(reference_dots)
        erased_reference += sum(replay.reference_erasure_mask)
        erased_candidate += sum(replay.candidate_erasure_mask)
        mask_mismatch_rows += replay.erasure_mask_mismatch_count > 0
        signature_mismatch_rows += replay.signature_mismatch_count > 0
        exact_rows += replay.exact
    if row_count == 0:
        raise ValueError("margin probe requires at least one row")
    return MarginProbeSummary(
        row_count=row_count,
        bit_count=bit_count,
        erased_reference_bits=erased_reference,
        erased_candidate_bits=erased_candidate,
        erasure_mask_mismatch_rows=mask_mismatch_rows,
        signature_mismatch_rows=signature_mismatch_rows,
        exact_rows=exact_rows,
    )


_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "secret_key",
        "private_key",
        "raw_key",
        "key_fingerprint",
        "key_sha256",
        "secret_fingerprint",
        "secret_sha256",
    }
)


def _validate_public_value(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("public artifact keys must be strings")
            if key.lower() in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"forbidden secret metadata at {path}.{key}")
            _validate_public_value(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_public_value(child, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"public artifact contains non-finite float at {path}")


def write_public_probe_artifact(path: str | Path, payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("public probe payload must be a mapping")
    normalized = dict(payload)
    existing_schema = normalized.get("schema_version")
    if existing_schema not in (None, PROBE_SCHEMA_VERSION):
        raise ValueError("public probe schema version mismatch")
    normalized["schema_version"] = PROBE_SCHEMA_VERSION
    _validate_public_value(normalized, path="$")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_candidate_pool(
    records: Iterable[CandidateRecord],
    *,
    retry: int,
) -> tuple[CandidateRecord, ...]:
    if isinstance(retry, bool) or not isinstance(retry, int) or retry <= 0:
        raise ValueError("retry must be a positive integer")
    items = tuple(records)
    if not items:
        raise ValueError("candidate pool must not be empty")
    grouped: dict[str, list[CandidateRecord]] = defaultdict(list)
    task_order: list[str] = []
    for item in items:
        if not isinstance(item.task_id, str) or not item.task_id:
            raise ValueError("candidate task ID must be non-empty")
        actual_sha256 = hashlib.sha256(item.final_code.encode("utf-8")).hexdigest()
        if not _SHA256_PATTERN.fullmatch(item.final_code_sha256) or actual_sha256 != item.final_code_sha256:
            raise ValueError("candidate final-code SHA-256 mismatch")
        if item.task_id not in grouped:
            task_order.append(item.task_id)
        grouped[item.task_id].append(item)
    for task_id in task_order:
        task_items = grouped[task_id]
        if len(task_items) != retry:
            raise ValueError(f"task {task_id} must contain exactly {retry} candidates")
        indices = tuple(item.attempt_index for item in task_items)
        if indices != tuple(range(retry)):
            raise ValueError(f"task {task_id} attempt indices must be ordered 0..{retry - 1}")
    return items


@dataclass(frozen=True)
class SelectionCapacity:
    task_count: int
    candidate_count: int
    positive_delta_tasks: int
    mean_delta: float
    maximum_delta_share: float
    selected_attempts: tuple[int, ...]
    per_task_deltas: tuple[float, ...]


def selection_capacity(
    records: Iterable[CandidateRecord],
    *,
    retry: int,
    score: Callable[[CandidateRecord], float],
) -> SelectionCapacity:
    items = validate_candidate_pool(records, retry=retry)
    grouped: dict[str, list[CandidateRecord]] = defaultdict(list)
    task_order: list[str] = []
    for item in items:
        if item.task_id not in grouped:
            task_order.append(item.task_id)
        grouped[item.task_id].append(item)
    deltas: list[float] = []
    selected_attempts: list[int] = []
    for task_id in task_order:
        task_items = grouped[task_id]
        scores = [float(score(item)) for item in task_items]
        if not all(math.isfinite(value) for value in scores):
            raise ValueError("candidate scores must be finite")
        selected_offset = max(range(len(scores)), key=scores.__getitem__)
        delta = scores[selected_offset] - scores[0]
        deltas.append(delta)
        selected_attempts.append(task_items[selected_offset].attempt_index)
    total_positive = sum(value for value in deltas if value > 0)
    maximum_share = (
        max((value for value in deltas if value > 0), default=0.0) / total_positive
        if total_positive > 0
        else 0.0
    )
    return SelectionCapacity(
        task_count=len(task_order),
        candidate_count=len(items),
        positive_delta_tasks=sum(value > 0 for value in deltas),
        mean_delta=sum(deltas) / len(deltas),
        maximum_delta_share=maximum_share,
        selected_attempts=tuple(selected_attempts),
        per_task_deltas=tuple(deltas),
    )


def _candidate_pool_sha256(records: Sequence[CandidateRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.task_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.attempt_index.to_bytes(4, "big", signed=False))
        digest.update(bytes.fromhex(record.final_code_sha256))
    return digest.hexdigest()


@dataclass(frozen=True)
class StructuralPoolProbe:
    capacity: SelectionCapacity
    input_pool_sha256: str
    output_pool_sha256: str
    candidate_pool_match_rate: float
    eligible_candidates: int
    eligible_task_count: int
    total_independent_units: int
    r3_input_fields: tuple[str, ...] = ("final_code",)


def structural_pool_capacity(
    records: Iterable[CandidateRecord],
    secret: ProbeSecret,
    *,
    retry: int,
    bit_count: int,
    minimum_independent_units: int,
) -> StructuralPoolProbe:
    items = validate_candidate_pool(records, retry=retry)
    scores = {
        (item.task_id, item.attempt_index): score_structural_code(
            item.final_code,
            secret,
            bit_count=bit_count,
            minimum_independent_units=minimum_independent_units,
        )
        for item in items
    }
    grouped: dict[str, list[CandidateRecord]] = defaultdict(list)
    task_order: list[str] = []
    for item in items:
        if item.task_id not in grouped:
            task_order.append(item.task_id)
        grouped[item.task_id].append(item)
    deltas: list[float] = []
    selected_attempts: list[int] = []
    eligible_task_count = 0
    for task_id in task_order:
        task_items = grouped[task_id]
        eligible_items = [
            item
            for item in task_items
            if scores[(item.task_id, item.attempt_index)].eligible
        ]
        if eligible_items:
            eligible_task_count += 1
            selected = max(
                eligible_items,
                key=lambda item: (
                    scores[(item.task_id, item.attempt_index)].score,
                    -item.attempt_index,
                ),
            )
            delta = (
                scores[(selected.task_id, selected.attempt_index)].score
                - scores[(task_items[0].task_id, task_items[0].attempt_index)].score
            )
        else:
            selected = task_items[0]
            delta = 0.0
        selected_attempts.append(selected.attempt_index)
        deltas.append(delta)
    positive_total = sum(value for value in deltas if value > 0)
    capacity = SelectionCapacity(
        task_count=len(task_order),
        candidate_count=len(items),
        positive_delta_tasks=sum(value > 0 for value in deltas),
        mean_delta=sum(deltas) / len(deltas),
        maximum_delta_share=(
            max((value for value in deltas if value > 0), default=0.0)
            / positive_total
            if positive_total > 0
            else 0.0
        ),
        selected_attempts=tuple(selected_attempts),
        per_task_deltas=tuple(deltas),
    )
    input_sha256 = _candidate_pool_sha256(items)
    output_items = tuple(items)
    output_sha256 = _candidate_pool_sha256(output_items)
    matched = sum(
        left == right for left, right in zip(items, output_items, strict=True)
    )
    return StructuralPoolProbe(
        capacity=capacity,
        input_pool_sha256=input_sha256,
        output_pool_sha256=output_sha256,
        candidate_pool_match_rate=matched / len(items),
        eligible_candidates=sum(score.eligible for score in scores.values()),
        eligible_task_count=eligible_task_count,
        total_independent_units=sum(
            score.independent_units for score in scores.values()
        ),
    )
