import shlex
import subprocess
import threading
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import (
    ALERT_INTERVAL,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    FILTER_LAYERS,
    JPEG_QUALITY,
    MAX_PERSON_ASPECT,
    MIN_PERSON_AREA_RATIO,
    MIN_PERSON_ASPECT,
    MIN_PERSON_HEIGHT_RATIO,
    MODE,
    RPICAM_COMMAND,
    RPICAM_EXTRA_ARGS,
    RTSP_URL,
)
from services.detector import Detector
from services.geometry import bbox_intersects_polygon
from services.notifier import HidNotifier
from services.zone_store import load_zone, save_zone


class VideoState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.zone = load_zone()
        self.filter_layers = dict(FILTER_LAYERS)
        self.detector = Detector()
        self.hid_notifier = HidNotifier()
        self.camera_error: Optional[str] = None
        self.capture_ready = False
        self.raw_frame: Optional[np.ndarray] = None
        self.raw_frame_id = 0
        self.frame: Optional[np.ndarray] = None
        self.encoded_frame: Optional[bytes] = None
        self.detections: List[Dict[str, List[float]]] = []
        self.collision = False
        self.alert_seq = 0
        self.last_alert_at = 0.0
        self.fps = 0.0
        self.capture_fps = 0.0
        self.frame_size = {"width": CAMERA_WIDTH, "height": CAMERA_HEIGHT}
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.capture_thread.start()
        self.process_thread.start()

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            self.camera_error = f"{MODE} camera unavailable"
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.camera_error = None
        self.capture_ready = True
        return cap

    def _capture_loop(self) -> None:
        if MODE == "RPICAM":
            self._rpicam_capture_loop()
            return
        self._ipcam_capture_loop()

    def _publish_capture_frame(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        with self.lock:
            self.raw_frame = frame
            self.raw_frame_id += 1
            self.frame_size = {"width": width, "height": height}
            if not self.detector.ready:
                self.frame = frame
                self.detections = []

        if not self.detector.ready:
            self._encode_frame(frame)

    def _ipcam_capture_loop(self) -> None:
        cap: Optional[cv2.VideoCapture] = None
        frames = 0
        fps_started = time.monotonic()

        while self.running:
            if cap is None:
                cap = self._open_camera()
                if cap is None:
                    self._publish_error_frame()
                    time.sleep(1)
                    continue

            ok, frame = cap.read()
            if not ok:
                self.camera_error = "camera read failed"
                self.capture_ready = False
                cap.release()
                cap = None
                self._publish_error_frame()
                time.sleep(0.5)
                continue

            self._publish_capture_frame(frame)

            frames += 1
            elapsed = time.monotonic() - fps_started
            if elapsed >= 1.0:
                with self.lock:
                    self.capture_fps = frames / elapsed
                frames = 0
                fps_started = time.monotonic()

    def _rpicam_args(self) -> List[str]:
        args = [
            RPICAM_COMMAND,
            "--codec",
            "mjpeg",
            "--width",
            str(CAMERA_WIDTH),
            "--height",
            str(CAMERA_HEIGHT),
            "--framerate",
            str(CAMERA_FPS),
            "--timeout",
            "0",
            "--nopreview",
            "--output",
            "-",
        ]
        if RPICAM_EXTRA_ARGS:
            args.extend(shlex.split(RPICAM_EXTRA_ARGS))
        return args

    def _rpicam_capture_loop(self) -> None:
        frames = 0
        fps_started = time.monotonic()
        buffer = bytearray()

        while self.running:
            try:
                process = subprocess.Popen(
                    self._rpicam_args(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,
                )
            except Exception as exc:
                self.camera_error = f"rpicam-vid failed: {exc}"
                self.capture_ready = False
                self._publish_error_frame()
                time.sleep(1)
                continue

            self.camera_error = None
            self.capture_ready = True

            try:
                while self.running and process.stdout:
                    chunk = process.stdout.read(4096)
                    if not chunk:
                        break
                    buffer.extend(chunk)

                    while True:
                        start = buffer.find(b"\xff\xd8")
                        end = buffer.find(b"\xff\xd9", start + 2)
                        if start < 0 or end < 0:
                            if len(buffer) > 1024 * 1024:
                                buffer.clear()
                            break

                        jpeg = bytes(buffer[start : end + 2])
                        del buffer[: end + 2]
                        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is None:
                            continue

                        self._publish_capture_frame(frame)
                        frames += 1
                        elapsed = time.monotonic() - fps_started
                        if elapsed >= 1.0:
                            with self.lock:
                                self.capture_fps = frames / elapsed
                            frames = 0
                            fps_started = time.monotonic()
            finally:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()

            self.camera_error = "rpicam-vid stopped"
            self.capture_ready = False
            self._publish_error_frame()
            time.sleep(0.5)

    def _process_loop(self) -> None:
        frames = 0
        fps_started = time.monotonic()
        last_processed_id = -1

        while self.running:
            with self.lock:
                frame_id = self.raw_frame_id
                frame = None if self.raw_frame is None else self.raw_frame.copy()

            if frame is None or frame_id == last_processed_id:
                time.sleep(0.01)
                continue

            last_processed_id = frame_id
            with self.lock:
                zone = list(self.zone)
                filter_layers = dict(self.filter_layers)

            detections = self.detector.detect(frame, filter_layers)
            collision = any(bbox_intersects_polygon(det["bbox"], zone) for det in detections)
            now = time.monotonic()

            with self.lock:
                self.collision = collision
                if collision and now - self.last_alert_at >= ALERT_INTERVAL:
                    self.alert_seq += 1
                    self.last_alert_at = now
                self.detections = detections
                self.frame = frame

            self.hid_notifier.send_state(collision)
            self._encode_frame(frame)

            frames += 1
            elapsed = time.monotonic() - fps_started
            if elapsed >= 1.0:
                with self.lock:
                    self.fps = frames / elapsed
                frames = 0
                fps_started = time.monotonic()

    def _encode_frame(self, frame: np.ndarray) -> None:
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            with self.lock:
                self.encoded_frame = buffer.tobytes()

    def _publish_error_frame(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        message = self.camera_error or "camera unavailable"
        cv2.putText(frame, message, (32, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
        self._encode_frame(frame)

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            return {
                "zone": self.zone,
                "detections": self.detections,
                "collision": self.collision,
                "alert_seq": self.alert_seq,
                "fps": round(self.fps, 1),
                "capture_fps": round(self.capture_fps, 1),
                "frame_size": self.frame_size,
                "camera_error": self.camera_error,
                "mode": MODE,
                "capture_ready": self.capture_ready,
                "detector_ready": self.detector.ready,
                "detector_error": self.detector.error,
                "raw_detection_count": self.detector.raw_detection_count,
                "filtered_detection_count": self.detector.filtered_detection_count,
                "filter_layers": self.filter_layers,
                "filter_values": {
                    "min_area_ratio": MIN_PERSON_AREA_RATIO,
                    "min_height_ratio": MIN_PERSON_HEIGHT_RATIO,
                    "min_aspect": MIN_PERSON_ASPECT,
                    "max_aspect": MAX_PERSON_ASPECT,
                },
                "hid": self.hid_notifier.snapshot(),
            }

    def update_zone(self, points: List[Dict[str, float]]) -> None:
        with self.lock:
            self.zone = points
            self.collision = False
            self.last_alert_at = 0.0
        save_zone(points)

    def update_filter_layers(self, filter_layers: Dict[str, object]) -> Dict[str, bool]:
        with self.lock:
            for key in self.filter_layers:
                if key in filter_layers:
                    self.filter_layers[key] = bool(filter_layers[key])
            return dict(self.filter_layers)
