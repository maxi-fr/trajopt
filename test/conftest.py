import sys
from pathlib import Path
from typing import Any

import pytest

# Ensure src/ package directory and test directory are on Python path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))
test_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(test_dir))

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


@pytest.fixture(scope="session")
def jl_altro() -> Any:
    """Initializes Julia and loads the vendored Altro.jl + TrajectoryOptimization.jl pair.

    Separate from `jl_to` (overview finding K / ticket 25): `altro_cross_jl/` is its own Julia
    environment that `Pkg.develop`s both `trajopt_jl/` and `altro_jl/`, so `jl_to`'s existing
    cross tests do not pay Altro's extra precompile cost.
    """
    try:
        from juliacall import Main as jl  # ty: ignore[unresolved-import]

        repo_root = Path(__file__).resolve().parent.parent
        jl_path = (repo_root / "altro_cross_jl").as_posix()
        jl.seval(f'using Pkg; Pkg.activate("{jl_path}")')
        jl.seval("using Altro, TrajectoryOptimization; const TO = TrajectoryOptimization")
    except Exception as e:
        pytest.skip(f"Julia runtime / juliacall / Altro unavailable: {e}")
    else:
        return jl
