from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ai4neb")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"


def manage_RM(*args, **kwargs):
    from .Regressor.RegressionModel import manage_RM as _manage_RM
    return _manage_RM(*args, **kwargs)
