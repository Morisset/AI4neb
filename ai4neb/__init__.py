from importlib.metadata import version, PackageNotFoundError
from pathlib import Path


def _read_version():
    # Prefer the live value in pyproject.toml when running from a source /
    # editable checkout, so a version bump is reflected without reinstalling.
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # backport for 3.8-3.10
        except ModuleNotFoundError:
            tomllib = None
    if tomllib is not None:
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        try:
            with pyproject.open("rb") as f:
                project = tomllib.load(f).get("project", {})
            if project.get("name") == "ai4neb" and "version" in project:
                return project["version"]
        except OSError:
            pass
    # Installed distribution metadata (wheel / non-editable install, where
    # pyproject.toml is not shipped alongside the package).
    try:
        return version("ai4neb")
    except PackageNotFoundError:
        return "0.0.0"


__version__ = _read_version()


def manage_RM(*args, **kwargs):
    from .Regressor.RegressionModel import manage_RM as _manage_RM
    return _manage_RM(*args, **kwargs)
