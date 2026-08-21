"""Pytest fixtures and configuration for unit and cross-verification tests."""

import sys
from pathlib import Path
from typing import Any

import pytest

# Ensure src/ package directory is on Python path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

import trajopt._env


@pytest.fixture(scope="session")
def jl_to() -> Any:
    """Initializes Julia and loads local TrajectoryOptimization.jl package."""
    try:
        from juliacall import Main as jl  # ty: ignore[unresolved-import]

        repo_root = Path(__file__).resolve().parent.parent
        jl_path = (repo_root / "trajopt_jl").as_posix()
        jl.seval(f'using Pkg; Pkg.activate("{jl_path}")')
        jl.seval("using TrajectoryOptimization; const TO = TrajectoryOptimization")
    except Exception as e:
        pytest.skip(f"Julia runtime / juliacall unavailable: {e}")
    else:
        return jl
