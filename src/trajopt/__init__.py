# Registers the Ipopt DLL directory on Windows at import time, reading IPOPT_DIR / IPOPT_BIN_DIR
# from the environment. Do not remove: without it, `import trajopt.transcription.ipopt` breaks
# Ipopt for Windows users in a way that will not be obvious to whoever "cleans this up" next.
from trajopt import _env as _env
