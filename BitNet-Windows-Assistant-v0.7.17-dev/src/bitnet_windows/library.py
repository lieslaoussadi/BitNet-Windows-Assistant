"""Safely initialize an explicitly selected local library."""
from __future__ import annotations

import os
from pathlib import Path


LIBRARY_DIRS = (
    "Library/downloads",
    "Library/packages",
    "Library/runtimes",
    "Library/models",
    "Library/manifests",
    "Library/quarantine",
    "Library/temp",
    "logs",
    "config",
)


class UnsafeLibraryRootError(OSError):
    """Raised when the selected library location is unsafe."""


def _is_link_or_junction(path: Path) -> bool:
    """Conservatively detect symbolic links and Windows junctions."""
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction()) if callable(is_junction) else False
    except OSError:
        # If a path cannot be inspected reliably, refuse it rather than guess.
        return True


def _git_checkout_ancestor(root: Path) -> Path | None:
    """Return a Git checkout containing root, if one exists."""
    current = root
    while True:
        marker = current / ".git"
        if marker.exists() or marker.is_symlink():
            return current
        if current.parent == current:
            return None
        current = current.parent


def initialize_library(root: Path) -> list[Path]:
    """Create library folders after validating the requested layout."""
    requested_root = root.expanduser()
    # Normalize the path without first resolving junctions/symlinks, so those
    # components can be detected before following them.
    requested_root = Path(os.path.abspath(os.fspath(requested_root)))

    # Check all existing components in the requested root path, not only the
    # final component. This also catches directory junctions in parent paths.
    for component in reversed((requested_root, *requested_root.parents)):
        if component.exists() or component.is_symlink():
            if _is_link_or_junction(component):
                raise UnsafeLibraryRootError(
                    f"Refusing a symbolic link or junction in library root: {component}"
                )

    root = requested_root.resolve()
    if not root.anchor or root == Path(root.anchor).resolve():
        raise UnsafeLibraryRootError(
            f"Refusing to initialize a library at a filesystem root: {root}"
        )

    if root.exists() and not root.is_dir():
        raise UnsafeLibraryRootError(
            f"The selected library root is not a directory: {root}"
        )

    git_root = _git_checkout_ancestor(root)
    if git_root is not None:
        raise UnsafeLibraryRootError(
            f"Refusing to initialize inside a Git checkout: {git_root}"
        )

    planned_paths = []
    # Validate every destination before creating any directory.
    for relative in LIBRARY_DIRS:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise UnsafeLibraryRootError(f"Unsafe relative library path: {relative}")

        candidate = root
        for part in relative_path.parts:
            candidate = candidate / part
            if _is_link_or_junction(candidate):
                raise UnsafeLibraryRootError(
                    f"Refusing a link or junction in the library layout: {candidate}"
                )
            if candidate.exists() and not candidate.is_dir():
                raise UnsafeLibraryRootError(
                    f"A file blocks a required directory: {candidate}"
                )
        planned_paths.append(root / relative_path)

    for path in planned_paths:
        path.mkdir(parents=True, exist_ok=True)

    return planned_paths