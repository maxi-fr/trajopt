import contextlib
import os
import sys
from pathlib import Path

import jax

__version__ = "0.1.0"

# Enable 64-bit precision by default across JAX
jax.config.update("jax_enable_x64", val=True)


def setup_ipopt_dlls() -> None:
    """On Windows, register Ipopt's DLL directory so cyipopt can load it at runtime.

    Reads ``IPOPT_DIR`` or ``IPOPT_BIN_DIR`` from the environment (set to the extracted COIN-OR
    Ipopt install; see the README), falling back to conventional install locations. A no-op on
    Linux and macOS, where Ipopt is found through the system library search.
    """
    if sys.platform != "win32":
        return

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


# Register Ipopt's DLL directory on Windows at import time.
setup_ipopt_dlls()
