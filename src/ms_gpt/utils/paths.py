"""Resolve external file paths from environment variables.

External paths consumed by this project can be overridden with environment
variables; repository-local released assets are used by default where applicable.

See README for the full list of variables and example values.
"""
from __future__ import annotations

import os
from pathlib import Path


class MissingPathError(RuntimeError):
    pass


def env_path(var: str, *, required: bool = True) -> Path | None:
    """Read a filesystem path from environment variable ``var``.

    If ``required`` is True and the variable is unset, raise a
    ``MissingPathError`` instructing the caller to set the variable. The path
    is not checked for existence here — callers decide when to materialise it.
    """
    val = os.environ.get(var)
    if not val:
        if required:
            raise MissingPathError(
                f"Environment variable {var} is not set. "
                f"See README.md for the list of required environment variables."
            )
        return None
    return Path(val).expanduser()


def env_path_or(var: str, fallback: str) -> Path:
    """Return ``env_path(var)`` if set, otherwise ``Path(fallback)``."""
    val = os.environ.get(var)
    return Path(val).expanduser() if val else Path(fallback)
