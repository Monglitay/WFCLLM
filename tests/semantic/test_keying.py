from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from wfcllm.semantic.keying import WatermarkKeying


def test_legacy_derive_outputs_remain_bit_for_bit_stable() -> None:
    keying = WatermarkKeying("deployment-secret", d=4)

    assert keying.derive("module", k=4, ordinal=None) == frozenset(
        {
            (0, 0, 0, 0),
            (0, 1, 0, 1),
            (0, 1, 1, 1),
            (1, 0, 0, 1),
        }
    )
    assert keying.derive("module", k=4, ordinal=0) == frozenset(
        {
            (0, 0, 0, 0),
            (0, 0, 0, 1),
            (0, 1, 0, 0),
            (1, 1, 1, 1),
        }
    )
    assert keying.derive(
        "function_definition",
        k=7,
        ordinal=3,
    ) == frozenset(
        {
            (0, 0, 0, 0),
            (0, 0, 1, 0),
            (0, 1, 0, 0),
            (0, 1, 1, 1),
            (1, 0, 0, 1),
            (1, 1, 0, 0),
            (1, 1, 1, 0),
        }
    )


def test_legacy_gamma_derive_output_remains_stable() -> None:
    keying = WatermarkKeying("deployment-secret", d=4, gamma=0.25)

    assert keying.derive("module") == frozenset(
        {
            (0, 0, 0, 0),
            (0, 1, 0, 1),
            (0, 1, 1, 1),
            (1, 0, 0, 1),
        }
    )


def test_descriptor_derivation_is_versioned_and_deterministic() -> None:
    keying = WatermarkKeying("deployment-secret", d=4)
    first = keying.derive_descriptor(
        contract_version="python-statement-window/v1",
        parent_descriptor=(
            "python-statement-window/v1|module/function_definition|"
            "parent=block|ordinal=1|role=body"
        ),
        k=4,
    )
    second = keying.derive_descriptor(
        contract_version="python-statement-window/v1",
        parent_descriptor=(
            "python-statement-window/v1|module/function_definition|"
            "parent=block|ordinal=1|role=body"
        ),
        k=4,
    )

    assert first == second
    assert first != keying.derive_descriptor(
        contract_version="python-statement-window/v1",
        parent_descriptor=(
            "python-statement-window/v1|module/function_definition|"
            "parent=block|ordinal=2|role=body"
        ),
        k=4,
    )
    assert first != keying.derive_descriptor(
        contract_version="python-statement-window/v2",
        parent_descriptor=(
            "python-statement-window/v1|module/function_definition|"
            "parent=block|ordinal=1|role=body"
        ),
        k=4,
    )


def test_descriptor_derivation_uses_exact_framed_unicode_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keying = WatermarkKeying("deployment-secret", d=4)
    calls: list[tuple[bytes, int]] = []

    def fake_derive(message: bytes, k: int) -> frozenset[tuple[int, ...]]:
        calls.append((message, k))
        return frozenset({(1, 0, 1, 0)})

    monkeypatch.setattr(keying, "_derive_from_message", fake_derive)

    result = keying.derive_descriptor(
        contract_version="窗口合同/v1",
        parent_descriptor="父节点|ordinal=一",
        k=4,
    )

    assert result == frozenset({(1, 0, 1, 0)})
    assert calls == [
        (
            "window-descriptor\0窗口合同/v1\0父节点|ordinal=一".encode(
                "utf-8"
            ),
            4,
        )
    ]


@pytest.mark.parametrize(
    ("contract_version", "parent_descriptor"),
    [
        (None, "descriptor"),
        (7, "descriptor"),
        ("", "descriptor"),
        ("contract", None),
        ("contract", ["descriptor"]),
        ("contract", ""),
        ("a", "\0b"),
        ("a\0", "b"),
    ],
)
def test_descriptor_derivation_rejects_invalid_or_ambiguous_components(
    contract_version: object,
    parent_descriptor: object,
) -> None:
    keying = WatermarkKeying("deployment-secret", d=4)

    with pytest.raises(ValueError):
        keying.derive_descriptor(
            contract_version=contract_version,
            parent_descriptor=parent_descriptor,
            k=4,
        )


@pytest.mark.parametrize(
    ("k", "error"),
    [
        (True, TypeError),
        (0, ValueError),
        (16, ValueError),
    ],
)
def test_descriptor_derivation_rejects_invalid_k(
    k: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        WatermarkKeying("deployment-secret", d=4).derive_descriptor(
            contract_version="python-statement-window/v1",
            parent_descriptor="descriptor",
            k=k,
        )


def test_descriptor_region_id_matches_exact_domain_separated_hmac_contract() -> None:
    secret = "deployment-secret"
    contract_version = "python-statement-window/v1"
    parent_descriptor = "合同|父节点|ordinal=1"
    allowed = frozenset(
        {
            (0, 0, 0, 0),
            (0, 1, 0, 1),
            (1, 0, 1, 0),
            (1, 1, 1, 1),
        }
    )
    canonical = json.dumps(
        {
            "allowed_signatures": sorted(allowed),
            "contract_version": contract_version,
            "d": 4,
            "k": 4,
            "parent_descriptor": parent_descriptor,
            "region_id_contract": "semantic-window-region/v1",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    message = b"semantic-window-region-id\0" + canonical
    expected_digest = hmac.new(
        secret.encode("utf-8"),
        message,
        hashlib.sha256,
    ).hexdigest()

    region_id = WatermarkKeying(secret, d=4).descriptor_region_id(
        contract_version=contract_version,
        parent_descriptor=parent_descriptor,
        k=4,
        allowed=allowed,
    )

    assert region_id == (
        "semantic-window-region/v1:hmac-sha256:" + expected_digest
    )
    assert hashlib.sha256(message).hexdigest() not in region_id


def test_descriptor_region_id_is_keyed_stable_and_input_specific() -> None:
    secret = "deployment-secret"
    descriptor = "python-statement-window/v1|module|parent=block|ordinal=1"
    allowed = frozenset(
        {
            (0, 0, 0, 0),
            (0, 1, 0, 1),
            (1, 0, 1, 0),
            (1, 1, 1, 1),
        }
    )
    changed_allowed = frozenset(
        {
            (0, 0, 0, 0),
            (0, 1, 0, 1),
            (1, 0, 1, 1),
            (1, 1, 1, 1),
        }
    )

    def region_id(
        keying: WatermarkKeying,
        *,
        parent_descriptor: str = descriptor,
        signatures: frozenset[tuple[int, ...]] = allowed,
    ) -> str:
        return keying.descriptor_region_id(
            contract_version="python-statement-window/v1",
            parent_descriptor=parent_descriptor,
            k=4,
            allowed=signatures,
        )

    keying = WatermarkKeying(secret, d=4)
    first = region_id(keying)

    assert first == region_id(keying)
    assert first != region_id(WatermarkKeying("other-secret", d=4))
    assert first != region_id(keying, parent_descriptor=descriptor + "/other")
    assert first != region_id(keying, signatures=changed_allowed)
    assert secret not in first
    assert descriptor not in first
    assert str(sorted(allowed)) not in first
