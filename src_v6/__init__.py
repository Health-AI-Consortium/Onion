# Re-export the algorithm entrypoints at package level so the vantage6 wrapper
# (wrap_algorithm: ``getattr(importlib.import_module(PKG_NAME), method_name)``)
# can dispatch "central" and "local" tasks.
from .central import central
from .local import local

__all__ = ["central", "local"]
