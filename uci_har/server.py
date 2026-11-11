




import importlib.machinery
import importlib.util
from pathlib import Path
import sys

_path = Path(__file__).parent / "bytecode" / f"server.{sys.implementation.cache_tag}.pyc"
if not _path.is_file():
    raise RuntimeError(
        "The supplied server bytecode requires CPython 3.12 or 3.9. "
        "Use Python 3.12 with this package, or replace this adapter with "
        "your original server.py source."
    )
_loader = importlib.machinery.SourcelessFileLoader("_original_wearfed_server", str(_path))
_spec = importlib.util.spec_from_loader(_loader.name, _loader)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_loader.name] = _module
_loader.exec_module(_module)
Server = _module.Server
