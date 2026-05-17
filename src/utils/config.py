from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

_ROOT = Path(__file__).parent.parent.parent
_cache: dict | None = None


def get_config() -> dict:
    global _cache
    if _cache is None:
        with open(_ROOT / "config.toml", "rb") as f:
            _cache = tomllib.load(f)
    return _cache
