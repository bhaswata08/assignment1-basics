import importlib.metadata

from jaxtyping import install_import_hook

hook = install_import_hook(
    modules="cs336_basics",
    # beartype does not apply the PEP 484 numeric tower by default, so a plain
    # `float` annotation would reject an `int` that every static checker accepts.
    typechecker="beartype.beartype(conf=beartype.BeartypeConf(is_pep484_tower=True))",
)
try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass
