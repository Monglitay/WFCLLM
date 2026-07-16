from __future__ import annotations

import os
import re
import stat
from pathlib import Path

_MAX_SECRET_BYTES = 1024 * 1024
_ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ENV_ENCODING_FAILURE = object()


def _strip_one_line_ending(value: bytes) -> bytes:
    if value.endswith(b"\r\n"):
        return value[:-2]
    if value.endswith(b"\n"):
        return value[:-1]
    return value


def _encode_environment_text(value: str) -> bytes | object:
    """Encode outside the caller's exception frame to avoid retaining value."""

    try:
        return value.encode("utf-8", errors="strict")
    except UnicodeError:
        return _ENV_ENCODING_FAILURE


def _reject_symlink_ancestors(path: Path) -> None:
    absolute_path = path if path.is_absolute() else Path.cwd() / path
    for ancestor in reversed(absolute_path.parent.parents):
        try:
            metadata = ancestor.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError("secret_file must be an existing regular file") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("secret_file must be an existing regular file")
    try:
        parent_metadata = absolute_path.parent.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError("secret_file must be an existing regular file") from exc
    if stat.S_ISLNK(parent_metadata.st_mode):
        raise ValueError("secret_file must be an existing regular file")


def _read_regular_file(path: Path) -> bytes:
    _reject_symlink_ancestors(path)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError("secret_file must be an existing regular file") from exc
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ValueError("secret_file must be an existing regular file")
    if metadata.st_size > _MAX_SECRET_BYTES:
        raise ValueError("secret_file is too large")

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("secret_file must be an existing regular file") from exc
    try:
        opened_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise ValueError("secret_file must be an existing regular file")
        if (
            opened_metadata.st_dev != metadata.st_dev
            or opened_metadata.st_ino != metadata.st_ino
        ):
            raise ValueError("secret_file changed while it was being opened")
        if opened_metadata.st_size > _MAX_SECRET_BYTES:
            raise ValueError("secret_file is too large")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            value = handle.read(_MAX_SECRET_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(value) > _MAX_SECRET_BYTES:
        raise ValueError("secret_file is too large")
    return value


def load_secret(
    *,
    secret_file: str | Path | None,
    env_name: str | None,
) -> bytes:
    """Load secret bytes from exactly one non-public runtime source."""

    if (secret_file is None) == (env_name is None):
        raise ValueError("exactly one of secret_file or env_name is required")

    if secret_file is not None:
        if isinstance(secret_file, bool) or not isinstance(secret_file, (str, Path)):
            raise ValueError("secret_file must be a filesystem path")
        try:
            path = Path(secret_file)
        except (TypeError, ValueError) as exc:
            raise ValueError("secret_file must be a filesystem path") from exc
        if not str(path) or "\x00" in str(path):
            raise ValueError("secret_file must be a filesystem path")
        value = _read_regular_file(path)
    else:
        if not isinstance(env_name, str) or _ENV_NAME_PATTERN.fullmatch(env_name) is None:
            raise ValueError("env_name must be a valid environment variable name")
        if getattr(os, "supports_bytes_environ", False):
            try:
                value = os.environb.get(os.fsencode(env_name))
            except (OSError, UnicodeError) as exc:
                raise ValueError("selected environment variable is unavailable") from exc
            if value is None:
                raise ValueError("selected environment variable is not set")
        else:
            environment_value = os.environ.get(env_name)
            if environment_value is None:
                raise ValueError("selected environment variable is not set")
            encoded_value = _encode_environment_text(environment_value)
            del environment_value
            if encoded_value is _ENV_ENCODING_FAILURE:
                raise ValueError(
                    "selected environment variable cannot be encoded safely"
                ) from None
            value = encoded_value

    if len(value) > _MAX_SECRET_BYTES:
        raise ValueError("selected secret value is too large")
    value = _strip_one_line_ending(value)
    if not value:
        raise ValueError("selected secret value is empty")
    return value
