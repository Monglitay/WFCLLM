from __future__ import annotations

import os
from pathlib import Path

import pytest

from wfcllm.common.secrets import load_secret


def test_secret_loader_file_returns_bytes_and_ignores_unselected_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "deployment.key"
    path.write_bytes(b"0101\n")
    monkeypatch.setenv("WFCLLM_DEPLOYMENT_KEY", "wrong")

    assert load_secret(secret_file=path, env_name=None) == b"0101"


def test_secret_loader_env_returns_utf8_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WFCLLM_DEPLOYMENT_KEY", "部署-key\n")
    assert load_secret(secret_file=None, env_name="WFCLLM_DEPLOYMENT_KEY") == "部署-key".encode()


@pytest.mark.skipif(
    not getattr(os, "supports_bytes_environ", False),
    reason="platform has no bytes environment",
)
def test_secret_loader_env_preserves_original_posix_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = b"WFCLLM_RAW_DEPLOYMENT_KEY"
    monkeypatch.setitem(os.environb, name, b"\xff\xfe\n")
    assert load_secret(secret_file=None, env_name=name.decode()) == b"\xff\xfe"


def test_secret_loader_non_posix_fallback_rejects_unencodable_value_without_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = "bad-\udcff-value"
    monkeypatch.setenv("WFCLLM_DEPLOYMENT_KEY", value)
    monkeypatch.setattr(os, "supports_bytes_environ", False)
    with pytest.raises(ValueError, match="environment") as exc_info:
        load_secret(secret_file=None, env_name="WFCLLM_DEPLOYMENT_KEY")
    assert value not in str(exc_info.value)
    assert value not in repr(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize(
    ("secret_file", "env_name"),
    [(None, None), ("key", "WFCLLM_DEPLOYMENT_KEY")],
)
def test_secret_loader_requires_exactly_one_source(
    secret_file: str | None,
    env_name: str | None,
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        load_secret(secret_file=secret_file, env_name=env_name)


@pytest.mark.parametrize("payload", [b"", b"\n", b"\r\n"])
def test_secret_loader_rejects_empty_file_value(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "deployment.key"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="empty"):
        load_secret(secret_file=path, env_name=None)


def test_secret_loader_removes_exactly_one_line_ending(tmp_path: Path) -> None:
    path = tmp_path / "deployment.key"
    path.write_bytes(b"value\n\n")
    assert load_secret(secret_file=path, env_name=None) == b"value\n"


def test_secret_loader_rejects_missing_or_empty_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFCLLM_DEPLOYMENT_KEY", raising=False)
    with pytest.raises(ValueError, match="not set"):
        load_secret(secret_file=None, env_name="WFCLLM_DEPLOYMENT_KEY")

    monkeypatch.setenv("WFCLLM_DEPLOYMENT_KEY", "")
    with pytest.raises(ValueError, match="empty"):
        load_secret(secret_file=None, env_name="WFCLLM_DEPLOYMENT_KEY")


def test_secret_loader_rejects_symlink_and_non_regular_file(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"value")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        load_secret(secret_file=link, env_name=None)

    with pytest.raises(ValueError, match="regular file"):
        load_secret(secret_file=tmp_path, env_name=None)


@pytest.mark.parametrize("ancestor_depth", [1, 2])
def test_secret_loader_rejects_any_existing_symlink_ancestor(
    tmp_path: Path,
    ancestor_depth: int,
) -> None:
    target = tmp_path / "real"
    target.mkdir()
    nested = target
    for index in range(ancestor_depth - 1):
        nested = nested / f"nested-{index}"
        nested.mkdir()
    (nested / "deployment.key").write_bytes(b"must-not-load")
    linked_ancestor = tmp_path / "linked"
    linked_ancestor.symlink_to(target, target_is_directory=True)
    secret_path = linked_ancestor.joinpath(
        *(f"nested-{index}" for index in range(ancestor_depth - 1)),
        "deployment.key",
    )

    with pytest.raises(ValueError, match="regular file"):
        load_secret(secret_file=secret_path, env_name=None)


def test_secret_loader_rejects_oversized_file(tmp_path: Path) -> None:
    path = tmp_path / "deployment.key"
    path.write_bytes(b"x" * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="too large"):
        load_secret(secret_file=path, env_name=None)


def test_secret_loader_does_not_include_secret_value_in_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = "must-not-leak"
    monkeypatch.setenv("WFCLLM_DEPLOYMENT_KEY", value)
    # An invalid source selection must fail without interpolating any source value.
    with pytest.raises(ValueError) as exc_info:
        load_secret(secret_file="key", env_name="WFCLLM_DEPLOYMENT_KEY")
    assert value not in str(exc_info.value)


def test_secret_module_does_not_define_a_logger() -> None:
    import wfcllm.common.secrets as secrets

    assert not any(name.lower() in {"logger", "log"} for name in vars(secrets))
    assert os.environ is not None
