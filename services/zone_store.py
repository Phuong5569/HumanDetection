import json
from typing import Dict, List

from config import CAMERA_HEIGHT, CAMERA_WIDTH, CONFIG_PATH


def default_zone(width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT) -> List[Dict[str, float]]:
    return [
        {"x": width * 0.25, "y": height * 0.30},
        {"x": width * 0.75, "y": height * 0.30},
        {"x": width * 0.78, "y": height * 0.78},
        {"x": width * 0.22, "y": height * 0.78},
    ]


def load_zone() -> List[Dict[str, float]]:
    if not CONFIG_PATH.exists():
        return default_zone()

    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        points = data.get("points", [])
        if len(points) != 4:
            return default_zone()
        return [{"x": float(p["x"]), "y": float(p["y"])} for p in points]
    except Exception:
        return default_zone()


def save_zone(points: List[Dict[str, float]]) -> None:
    CONFIG_PATH.write_text(
        json.dumps({"points": points}, indent=2),
        encoding="utf-8",
    )
