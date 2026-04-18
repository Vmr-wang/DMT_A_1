from __future__ import annotations

import importlib
from typing import Optional


class MissingDependencyError(ImportError):
    """Raised when an optional runtime dependency is not installed."""


def require_dependency(module_name: str, install_name: Optional[str] = None):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        package_name = install_name or module_name
        raise MissingDependencyError(
            f"Missing optional dependency '{module_name}'. Install '{package_name}' first."
        ) from exc
