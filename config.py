import os
from pathlib import Path
from urllib.parse import quote

# Must be set before OpenCV opens RTSP stream.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;udp|max_delay;0|fflags;nobuffer|flags;low_delay|reorder_queue_size;0",
)

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
CONFIG_PATH = BASE_DIR / "zone_config.json"


def load_env_file(path: Path = ENV_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    return values


ENV = load_env_file()


def env(key: str, default: str) -> str:
    return ENV.get(key, os.environ.get(key, default))


def env_int(key: str, default: str) -> int:
    return int(env(key, default))


def env_float(key: str, default: str) -> float:
    return float(env(key, default))


def env_bool(key: str, default: str) -> bool:
    return env(key, default).strip().lower() not in {"0", "false", "no", "off", "-1"}


def env_path(key: str, default: str) -> str:
    raw_path = Path(env(key, default))
    if raw_path.is_absolute():
        return str(raw_path)
    return str(BASE_DIR / raw_path)


def layer_enabled(value: float | bool) -> bool:
    return value != -1 and value is not False


MODEL_PATH = env_path("YOLO_MODEL", "models/yolov8n_ncnn_model")
INFER_SIZE = env_int("INFER_SIZE", "320")
CONFIDENCE = env_float("CONFIDENCE", "0.45")
IOU_THRESHOLD = env_float("IOU_THRESHOLD", "0.45")
MAX_DETECTIONS = env_int("MAX_DETECTIONS", "5")
MIN_PERSON_AREA_RATIO = env_float("MIN_PERSON_AREA_RATIO", "0.0005")
MIN_PERSON_HEIGHT_RATIO = env_float("MIN_PERSON_HEIGHT_RATIO", "0.025")
MIN_PERSON_ASPECT = env_float("MIN_PERSON_ASPECT", "0.08")
MAX_PERSON_ASPECT = env_float("MAX_PERSON_ASPECT", "2.2")
KEEP_BEST_PERSON_ONLY = env_bool("KEEP_BEST_PERSON_ONLY", "1")

FILTER_LAYERS = {
    "min_area": layer_enabled(MIN_PERSON_AREA_RATIO),
    "min_height": layer_enabled(MIN_PERSON_HEIGHT_RATIO),
    "min_aspect": layer_enabled(MIN_PERSON_ASPECT),
    "max_aspect": layer_enabled(MAX_PERSON_ASPECT),
    "keep_best": layer_enabled(KEEP_BEST_PERSON_ONLY),
}

CAMERA_WIDTH = env_int("CAMERA_WIDTH", "640")
CAMERA_HEIGHT = env_int("CAMERA_HEIGHT", "480")
CAMERA_FPS = env_int("CAMERA_FPS", "15")
JPEG_QUALITY = env_int("JPEG_QUALITY", "75")
ALERT_INTERVAL = env_float("ALERT_INTERVAL", "1.0")
HID_DEVICE = env("HID_DEVICE", "/dev/hidraw0")
HID_ENABLED = env_bool("HID_ENABLED", "1")
PORT = env_int("PORT", "5000")
MODE = env("MODE", "IPCAM").upper()

RTSP_URL = env("RTSP_URL", "")
if not RTSP_URL:
    CAMERA_IP = env("CAMERA_IP", "192.168.2.64")
    USERNAME = env("CAMERA_USERNAME", "admin")
    PASSWORD = env("CAMERA_PASSWORD", "")
    CHANNEL = env("CAMERA_CHANNEL", "102")
    auth = f"{quote(USERNAME)}:{quote(PASSWORD)}@" if USERNAME or PASSWORD else ""
    RTSP_URL = f"rtsp://{auth}{CAMERA_IP}:554/Streaming/Channels/{CHANNEL}"

RPICAM_COMMAND = env("RPICAM_COMMAND", "rpicam-vid")
RPICAM_EXTRA_ARGS = env("RPICAM_EXTRA_ARGS", "")

PERSON_CLASS_ID = 0
