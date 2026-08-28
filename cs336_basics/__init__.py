import importlib.metadata

from jaxtyping import install_import_hook

hook = install_import_hook(modules="cs336_basics", typechecker="beartype.beartype")
try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass
