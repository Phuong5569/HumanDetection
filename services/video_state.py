import shlex
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from config import (
    ALERT_INTERVAL,
    CAMERA_ANGLE,
    CAMERA_FLIP,
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
    RPICAM_CAMERA_COUNT,
    RPICAM_COMMAND,
    RPICAM_EXTRA_ARGS,
    RPICAM_LEFT_EXTRA_ARGS,
    RPICAM_RIGHT_EXTRA_ARGS,
    RTSP_URL,
    TEST_CAMERA_MAX_INDEX,
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
        self.raw_dual_frames: Optional[Dict[str, object]] = None
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

    def _open_test_camera(self) -> Optional[cv2.VideoCapture]:
        for index in range(TEST_CAMERA_MAX_INDEX + 1):
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            self.camera_error = None
            self.capture_ready = True
            return cap

        self.camera_error = f"{MODE} camera unavailable: no OpenCV camera found"
        return None

    def _capture_loop(self) -> None:
        if MODE == "RPICAM":
            self._rpicam_capture_loop()
            return
        if MODE == "TESTMODE":
            self._test_camera_capture_loop()
            return
        self._ipcam_capture_loop()

    def _orient_frame(self, frame: np.ndarray) -> np.ndarray:
        if CAMERA_ANGLE == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif CAMERA_ANGLE == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif CAMERA_ANGLE == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if CAMERA_FLIP == "h":
            frame = cv2.flip(frame, 1)
        elif CAMERA_FLIP == "v":
            frame = cv2.flip(frame, 0)
        elif CAMERA_FLIP == "hv":
            frame = cv2.flip(frame, -1)

        return frame

    def _publish_capture_frame(self, frame: np.ndarray) -> None:
        frame = self._orient_frame(frame)
        height, width = frame.shape[:2]
        with self.lock:
            self.raw_frame = frame
            self.raw_dual_frames = None
            self.raw_frame_id += 1
            self.frame_size = {"width": width, "height": height}
            if not self.detector.ready:
                self.frame = frame
                self.detections = []

        if not self.detector.ready:
            self._encode_frame(frame)

    def _publish_dual_capture_frame(
        self,
        frame: np.ndarray,
        left: Optional[np.ndarray],
        right: Optional[np.ndarray],
        left_width: int,
        pre_orient_width: int,
        pre_orient_height: int,
    ) -> None:
        frame = self._orient_frame(frame)
        height, width = frame.shape[:2]
        with self.lock:
            self.raw_frame = frame
            self.raw_dual_frames = {
                "left": None if left is None else left.copy(),
                "right": None if right is None else right.copy(),
                "left_width": left_width,
                "pre_orient_width": pre_orient_width,
                "pre_orient_height": pre_orient_height,
            }
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

    def _test_camera_capture_loop(self) -> None:
        cap: Optional[cv2.VideoCapture] = None
        frames = 0
        fps_started = time.monotonic()

        while self.running:
            if cap is None:
                cap = self._open_test_camera()
                if cap is None:
                    self._publish_error_frame()
                    time.sleep(1)
                    continue

            ok, frame = cap.read()
            if not ok:
                self.camera_error = "test camera read failed"
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

    def _rpicam_args(self, camera_extra_args: str = "") -> List[str]:
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
        if camera_extra_args:
            args.extend(shlex.split(camera_extra_args))
        return args

    def _read_rpicam_frames(
        self,
        camera_extra_args: str,
        label: str,
        on_frame: Callable[[np.ndarray], None],
        on_error: Callable[[Optional[str]], None],
    ) -> None:
        buffer = bytearray()

        while self.running:
            try:
                process = subprocess.Popen(
                    self._rpicam_args(camera_extra_args),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,
                )
            except Exception as exc:
                on_error(f"{label} rpicam-vid failed: {exc}")
                time.sleep(1)
                continue

            on_error(None)

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

                        on_frame(frame)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()

            on_error(f"{label} rpicam-vid stopped")
            time.sleep(0.5)

    def _rpicam_capture_loop(self) -> None:
        if RPICAM_CAMERA_COUNT == 2:
            self._dual_rpicam_capture_loop()
            return

        frames = 0
        fps_started = time.monotonic()

        def on_frame(frame: np.ndarray) -> None:
            nonlocal frames, fps_started
            self._publish_capture_frame(frame)
            frames += 1
            elapsed = time.monotonic() - fps_started
            if elapsed >= 1.0:
                with self.lock:
                    self.capture_fps = frames / elapsed
                frames = 0
                fps_started = time.monotonic()

        def on_error(error: Optional[str]) -> None:
            self.camera_error = error
            self.capture_ready = error is None
            if error:
                self._publish_error_frame()

        self._read_rpicam_frames("", "rpicam", on_frame, on_error)

    def _placeholder_frame(self, width: int, height: int, message: str) -> np.ndarray:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, message, (24, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        return frame

    def _prepare_rpicam_frames(
        self,
        left: Optional[np.ndarray],
        right: Optional[np.ndarray],
        left_error: Optional[str],
        right_error: Optional[str],
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        height = CAMERA_HEIGHT
        width = CAMERA_WIDTH
        if left is not None:
            height, width = left.shape[:2]
        elif right is not None:
            height, width = right.shape[:2]

        detect_left = left
        detect_right = right
        if left is None:
            left = self._placeholder_frame(width, height, left_error or "left camera waiting")
        if right is None:
            right = self._placeholder_frame(width, height, right_error or "right camera waiting")
        elif right.shape[0] != height:
            scale = height / right.shape[0]
            right_width = max(1, round(right.shape[1] * scale))
            right = cv2.resize(right, (right_width, height))
            detect_right = right

        return left, right, detect_left, detect_right

    def _merge_rpicam_frames(
        self,
        left: Optional[np.ndarray],
        right: Optional[np.ndarray],
        left_error: Optional[str],
        right_error: Optional[str],
    ) -> np.ndarray:
        left_frame, right_frame, _, _ = self._prepare_rpicam_frames(left, right, left_error, right_error)
        return np.hstack((left_frame, right_frame))

    def _dual_rpicam_capture_loop(self) -> None:
        latest_frames: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        latest_errors: Dict[str, Optional[str]] = {"left": "left camera waiting", "right": "right camera waiting"}
        latest_ids = {"left": 0, "right": 0}
        latest_lock = threading.Lock()
        frames = 0
        fps_started = time.monotonic()
        last_published_ids = (-1, -1)

        def make_on_frame(side: str) -> Callable[[np.ndarray], None]:
            def on_frame(frame: np.ndarray) -> None:
                with latest_lock:
                    latest_frames[side] = frame
                    latest_errors[side] = None
                    latest_ids[side] += 1

            return on_frame

        def make_on_error(side: str) -> Callable[[Optional[str]], None]:
            def on_error(error: Optional[str]) -> None:
                with latest_lock:
                    latest_errors[side] = error
                    latest_ids[side] += 1

            return on_error

        workers = [
            threading.Thread(
                target=self._read_rpicam_frames,
                args=(RPICAM_LEFT_EXTRA_ARGS, "left", make_on_frame("left"), make_on_error("left")),
                daemon=True,
            ),
            threading.Thread(
                target=self._read_rpicam_frames,
                args=(RPICAM_RIGHT_EXTRA_ARGS, "right", make_on_frame("right"), make_on_error("right")),
                daemon=True,
            ),
        ]
        for worker in workers:
            worker.start()

        while self.running:
            with latest_lock:
                left = None if latest_frames["left"] is None else latest_frames["left"].copy()
                right = None if latest_frames["right"] is None else latest_frames["right"].copy()
                left_error = latest_errors["left"]
                right_error = latest_errors["right"]
                frame_ids = (latest_ids["left"], latest_ids["right"])

            if frame_ids == last_published_ids and left is None and right is None:
                time.sleep(0.05)
                continue

            left_frame, right_frame, detect_left, detect_right = self._prepare_rpicam_frames(
                left, right, left_error, right_error
            )
            merged = np.hstack((left_frame, right_frame))
            errors = [error for error in (left_error, right_error) if error]
            self.camera_error = "; ".join(errors) if errors else None
            self.capture_ready = not errors
            self._publish_dual_capture_frame(
                merged,
                detect_left,
                detect_right,
                left_frame.shape[1],
                merged.shape[1],
                merged.shape[0],
            )
            last_published_ids = frame_ids

            frames += 1
            elapsed = time.monotonic() - fps_started
            if elapsed >= 1.0:
                with self.lock:
                    self.capture_fps = frames / elapsed
                frames = 0
                fps_started = time.monotonic()

            time.sleep(0.01)

    def _process_loop(self) -> None:
        frames = 0
        fps_started = time.monotonic()
        last_processed_id = -1

        while self.running:
            with self.lock:
                frame_id = self.raw_frame_id
                frame = None if self.raw_frame is None else self.raw_frame.copy()
                dual_frames = self.raw_dual_frames

            if frame is None or frame_id == last_processed_id:
                time.sleep(0.01)
                continue

            last_processed_id = frame_id
            with self.lock:
                zone = list(self.zone)
                filter_layers = dict(self.filter_layers)

            if dual_frames is None:
                detections = self.detector.detect(frame, filter_layers)
            else:
                detections = self._detect_dual_frames(dual_frames, filter_layers)
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

    def _detect_dual_frames(self, dual_frames: Dict[str, object], filter_layers: Dict[str, bool]) -> List[Dict[str, List[float]]]:
        detections: List[Dict[str, List[float]]] = []
        raw_count = 0
        filtered_count = 0
        left = dual_frames["left"]
        right = dual_frames["right"]
        left_width = int(dual_frames["left_width"])
        pre_orient_width = int(dual_frames["pre_orient_width"])
        pre_orient_height = int(dual_frames["pre_orient_height"])

        if left is not None:
            left_detections = self.detector.detect(left, filter_layers)
            raw_count += self.detector.raw_detection_count
            filtered_count += self.detector.filtered_detection_count
            detections.extend(left_detections)

        if right is not None:
            right_detections = self.detector.detect(right, filter_layers)
            raw_count += self.detector.raw_detection_count
            filtered_count += self.detector.filtered_detection_count
            for detection in right_detections:
                x1, y1, x2, y2 = detection["bbox"]
                detections.append(
                    {
                        "bbox": [x1 + left_width, y1, x2 + left_width, y2],
                        "confidence": detection["confidence"],
                    }
                )

        self.detector.raw_detection_count = raw_count
        self.detector.filtered_detection_count = filtered_count
        return [
            {
                "bbox": self._orient_bbox(det["bbox"], pre_orient_width, pre_orient_height),
                "confidence": det["confidence"],
            }
            for det in detections
        ]

    def _orient_bbox(self, bbox: List[float], width: int, height: int) -> List[float]:
        x1, y1, x2, y2 = bbox
        points = [
            self._orient_point(x1, y1, width, height),
            self._orient_point(x2, y1, width, height),
            self._orient_point(x2, y2, width, height),
            self._orient_point(x1, y2, width, height),
        ]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return [min(xs), min(ys), max(xs), max(ys)]

    def _orient_point(self, x: float, y: float, width: int, height: int) -> tuple[float, float]:
        oriented_width = width
        oriented_height = height
        if CAMERA_ANGLE == 90:
            x, y = height - y, x
            oriented_width, oriented_height = height, width
        elif CAMERA_ANGLE == 180:
            x, y = width - x, height - y
        elif CAMERA_ANGLE == 270:
            x, y = y, width - x
            oriented_width, oriented_height = height, width

        if CAMERA_FLIP in {"h", "hv"}:
            x = oriented_width - x
        if CAMERA_FLIP in {"v", "hv"}:
            y = oriented_height - y
        return x, y

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
