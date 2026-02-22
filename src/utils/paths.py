from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def ensure_parent(path: str | Path) -> Path:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load config files. Install it with `pip install pyyaml`."
        ) from exc

    cfg_path = resolve_path(path)
    with cfg_path.open("r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj) or {}
    return data

