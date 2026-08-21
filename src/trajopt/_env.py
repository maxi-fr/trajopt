"""Environment and DLL loader helper for TrajectoryOptimization (trajopt)."""

import contextlib
import os
import sys
from pathlib import Path


def load_env_file(dotenv_path: str | Path | None = None) -> None:
    """Load key-value pairs from a .env file into os.environ if not already present."""
    if dotenv_path is None:
        # Search current working directory and parent directories (up to 4 levels)
        cwd = Path.cwd()
        for p in [cwd, *cwd.parents[:4]]:
            candidate = p / ".env"
            if candidate.is_file():
                dotenv_path = candidate
                break

    if not dotenv_path or not Path(dotenv_path).is_file():
        return

    with (
        contextlib.suppress(OSError, UnicodeDecodeError),
        Path(dotenv_path).open(encoding="utf-8") as f,
    ):
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val


def setup_ipopt_dlls() -> None:
    """On Windows, ensure Ipopt DLL directories are added to Python's DLL search paths."""
    if sys.platform != "win32":
        return

    load_env_file()

    ipopt_dir = os.environ.get("IPOPT_DIR")
    search_dirs = [
        os.environ.get("IPOPT_BIN_DIR"),
        str(Path(ipopt_dir) / "bin") if ipopt_dir else None,
        r"C:\Ipopt\Ipopt-3.14.19-win64-msvs2022-md\bin",
        r"C:\Ipopt\bin",
    ]

    for directory in search_dirs:
        if directory and Path(directory).is_dir():
            with contextlib.suppress(OSError):
                os.add_dll_directory(directory)


# Automatically configure environment and DLL paths upon module import
setup_ipopt_dlls()
