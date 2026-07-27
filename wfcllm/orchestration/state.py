"""Concurrency-safe persistence for the WFCLLM phase run state."""
from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:  # Linux/macOS production path. Fail closed elsewhere instead of using an unsafe lock.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on platforms without flock(2)
    fcntl = None  # type: ignore[assignment]

# Phase name source of truth for a Fresh Reproduction Run.
PHASES = [
    "encoder",
    "gate-data",
    "gate-train",
    "generate",
    "calibrate",
    "detect",
    "report",
    "audit",
]
ALL_PHASES = PHASES

DEFAULT_STATE_FILE = Path("data/run_state.json")
_MAX_STATE_BYTES = 16 * 1024 * 1024


class RunStateManager:
    """Persistent phase tracker with strict validation and lost-update protection.

    Writers serialize through a sidecar ``.lock`` file, reload while holding the
    lock, merge their phase update, and publish through an fsynced atomic replace.
    Only the current eight-phase schema is accepted.
    """

    def __init__(self, path: Path = DEFAULT_STATE_FILE) -> None:
        self._path = Path(path)
        self._existed_at_initialization = self._path.exists()
        self._lock_path = self._path.with_name(f"{self._path.name}.lock")
        self._data: dict[str, dict[str, Any]] = (
            self._read_locked()
            if self._existed_at_initialization
            else self._default_data()
        )

    @property
    def existed_at_initialization(self) -> bool:
        """Whether this manager opened an already-persisted run state."""

        return self._existed_at_initialization

    @staticmethod
    def _default_data() -> dict[str, dict[str, Any]]:
        return {phase: {"done": False} for phase in ALL_PHASES}

    @staticmethod
    def _reject_symlink_path(path: Path) -> None:
        absolute = path if path.is_absolute() else Path.cwd() / path
        for candidate in (absolute, *absolute.parents):
            try:
                if candidate.is_symlink():
                    raise ValueError("run state path cannot traverse symlinks")
            except OSError as exc:
                raise ValueError("run state path cannot be safely inspected") from exc

    @contextmanager
    def _locked(self, *, exclusive: bool) -> Iterator[None]:
        if fcntl is None:  # pragma: no cover - explicit safe fallback on unsupported OSes
            raise RuntimeError("run state locking requires an operating system with flock(2)")
        self._reject_symlink_path(self._path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._reject_symlink_path(self._path)
        self._reject_symlink_path(self._lock_path)
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = -1
        try:
            descriptor = os.open(self._lock_path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        except OSError as exc:
            raise ValueError("run state lock is missing or unsafe") from exc
        finally:
            if descriptor >= 0:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)

    def _read_locked(self) -> dict[str, dict[str, Any]]:
        self._reject_symlink_path(self._path)
        if not self._path.exists():
            return self._default_data()
        with self._locked(exclusive=False):
            return self._load_unlocked()

    def _load_unlocked(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return self._default_data()
        self._reject_symlink_path(self._path)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = -1
        try:
            descriptor = os.open(self._path, flags)
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_STATE_BYTES:
                raise ValueError("run state must be a bounded regular file")
            payload = bytearray()
            while chunk := os.read(descriptor, min(1024 * 1024, _MAX_STATE_BYTES + 1 - len(payload))):
                payload.extend(chunk)
                if len(payload) > _MAX_STATE_BYTES:
                    raise ValueError("run state exceeds the size limit")
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("run state changed while reading")
        except OSError as exc:
            raise ValueError("run state file is missing or unsafe") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

        def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("run state contains a duplicate key")
                result[key] = value
            return result

        def no_constants(value: str) -> None:
            raise ValueError(f"run state contains non-finite number {value}")

        try:
            value = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=no_duplicates,
                parse_constant=no_constants,
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("run state is invalid JSON") from exc
        return self._validate(value)

    @classmethod
    def _validate(cls, value: object) -> dict[str, dict[str, Any]]:
        if not isinstance(value, dict):
            raise ValueError("run state root must be an object")
        if set(value) != set(ALL_PHASES):
            raise ValueError(
                "run state must contain exactly the current eight phases"
            )
        result: dict[str, dict[str, Any]] = {}
        for phase, row in value.items():
            if not isinstance(phase, str) or phase not in ALL_PHASES:
                raise ValueError("run state contains an unknown phase")
            if not isinstance(row, dict):
                raise ValueError("run state phase row must be an object")
            if not isinstance(row.get("done"), bool):
                raise ValueError("run state phase done flag must be boolean")
            cls._validate_json_value(row)
            result[phase] = dict(row)
        return result

    @classmethod
    def _validate_json_value(cls, value: object) -> None:
        if value is None or isinstance(value, (str, bool, int)):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("run state contains a non-finite number")
            return
        if isinstance(value, list):
            for item in value:
                cls._validate_json_value(item)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("run state object keys must be strings")
                cls._validate_json_value(item)
            return
        raise ValueError("run state contains a non-JSON value")

    def _save_unlocked(self, data: dict[str, dict[str, Any]]) -> None:
        validated = self._validate(data)
        try:
            payload = (
                json.dumps(validated, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ValueError("run state cannot be encoded as canonical JSON") from exc
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self._path.name}.",
            suffix=".tmp",
            dir=self._path.parent,
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                descriptor = -1
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._path)
            os.chmod(self._path, 0o600)
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_descriptor = os.open(self._path.parent, directory_flags)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def is_done(self, phase: str) -> bool:
        self._data = self._read_locked()
        return self._data.get(phase, {}).get("done", False) is True

    def get(self, phase: str, key: str) -> Any | None:
        self._data = self._read_locked()
        return self._data.get(phase, {}).get(key)

    def mark_done(self, phase: str, **kwargs: Any) -> None:
        if phase not in ALL_PHASES:
            raise ValueError("run state contains an unknown phase")
        if "done" in kwargs or "completed_at" in kwargs:
            raise ValueError("run state completion metadata is reserved")
        with self._locked(exclusive=True):
            data = self._load_unlocked()
            data[phase] = {
                "done": True,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                **kwargs,
            }
            self._save_unlocked(data)
            self._data = data

    def status(self) -> dict[str, dict[str, Any]]:
        self._data = self._read_locked()
        return {
            phase: dict(self._data.get(phase, {"done": False}))
            for phase in ALL_PHASES
        }
