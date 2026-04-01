import importlib
import pkgutil


def discover_tools():
    """Yield (blueprint, url_prefix, meta) for every tool package."""
    import tools
    for _importer, modname, ispkg in pkgutil.iter_modules(tools.__path__):
        if not ispkg:
            continue
        mod = importlib.import_module(f"tools.{modname}")
        bp = getattr(mod, "blueprint", None)
        meta = getattr(mod, "TOOL_META", None)
        if bp and meta:
            yield bp, meta["url"], meta
