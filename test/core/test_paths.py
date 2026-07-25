import os
from pathlib import Path

import pytest

from core.paths import get_root_dir


def test_get_root_dir_raises_when_no_pyproject_above(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """get_root_dir used to raise IndexError at filesystem root; must now raise a clear RuntimeError."""
    isolated: Path = tmp_path / "no_pyproject_here"
    isolated.mkdir()
    monkeypatch.chdir(isolated)

    # Sanity check: the tmp path must not have a pyproject.toml on any ancestor for this test to
    # be meaningful. tmp_path fixtures live under /tmp on Linux which satisfies this.
    walker: Path = isolated
    while walker != walker.parent:
        assert not walker.joinpath("pyproject.toml").exists(), f"unexpected pyproject at {walker}"
        walker = walker.parent

    with pytest.raises(RuntimeError, match="pyproject.toml not found"):
        get_root_dir()


def test_get_root_dir_finds_project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """When run from anywhere under the project, get_root_dir returns the folder with pyproject.toml."""
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    assert project_root.joinpath("pyproject.toml").exists()

    monkeypatch.chdir(project_root)
    assert get_root_dir() == project_root

    # Also works from a nested directory.
    nested: Path = project_root / "core"
    if nested.exists():
        monkeypatch.chdir(nested)
        assert get_root_dir() == project_root

    # Reset for cleanliness even though monkeypatch will undo cwd.
    os.chdir(project_root)
