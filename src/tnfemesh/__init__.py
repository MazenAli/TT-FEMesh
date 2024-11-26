from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tnfemesh")
except PackageNotFoundError:
    __version__ = "unknown"
