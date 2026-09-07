from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import (
    CONFIDENCE,
    INFER_SIZE,
    IOU_THRESHOLD,
    MAX_DETECTIONS,
    MAX_PERSON_ASPECT,
    MIN_PERSON_AREA_RATIO,
    MIN_PERSON_ASPECT,
    MIN_PERSON_HEIGHT_RATIO,
    MODEL_PATH,
    PERSON_CLASS_ID,
    YOLO_DEVICE,
)
from services.geometry import nms_detections

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - handled at runtime for friendly UI status
    YOLO = None


class Detector:
    def __init__(self) -> None:
        self.model = None
        self.error: Optional[str] = None
        self.ready = False
        self.raw_detection_count = 0
        self.filtered_detection_count = 0
        self._load()

    def _load(self) -> None:
        if YOLO is None:
            self.error = "ultralytics package not installed"
            return
        model_path = Path(MODEL_PATH)
        is_bare_model_name = model_path.name == MODEL_PATH
        if not is_bare_model_name and not model_path.exists():
            self.error = f"model not found: {MODEL_PATH}"
            return
        try:
            self.model = YOLO(MODEL_PATH)
            self.ready = True
            self.error = None
        except Exception as exc:
            self.error = str(exc)

    def detect(self, frame: np.ndarray, filter_layers: Dict[str, bool]) -> List[Dict[str, List[float]]]:
        if not self.ready or self.model is None:
            return []

        frame_height, frame_width = frame.shape[:2]
        min_area = frame_width * frame_height * MIN_PERSON_AREA_RATIO
        min_height = frame_height * MIN_PERSON_HEIGHT_RATIO
        results = self.model.predict(
            frame,
            imgsz=INFER_SIZE,
            conf=CONFIDENCE,
            iou=IOU_THRESHOLD,
            classes=[PERSON_CLASS_ID],
            max_det=MAX_DETECTIONS,
            device=YOLO_DEVICE or None,
            verbose=False,
        )

        detections: List[Dict[str, List[float]]] = []
        raw_count = 0
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                raw_count += 1
                xyxy = box.xyxy[0].cpu().numpy().astype(float).tolist()
                x1, y1, x2, y2 = xyxy
                box_width = max(0.0, x2 - x1)
                box_height = max(0.0, y2 - y1)
                aspect = box_width / box_height if box_height > 0 else 0.0
                if filter_layers["min_area"] and box_width * box_height < min_area:
                    continue
                if filter_layers["min_height"] and box_height < min_height:
                    continue
                if filter_layers["min_aspect"] and aspect < MIN_PERSON_ASPECT:
                    continue
                if filter_layers["max_aspect"] and aspect > MAX_PERSON_ASPECT:
                    continue
                conf = float(box.conf[0].cpu().numpy())
                detections.append({"bbox": xyxy, "confidence": conf})

        detections = nms_detections(detections)
        self.raw_detection_count = raw_count
        self.filtered_detection_count = len(detections)
        if filter_layers["keep_best"] and detections:
            return [detections[0]]
        return detections
