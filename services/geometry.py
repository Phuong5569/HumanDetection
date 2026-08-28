from typing import Dict, List, Tuple

from config import IOU_THRESHOLD, MAX_DETECTIONS


def point_in_poly(point: Tuple[float, float], poly: List[Tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def ccw(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
    d: Tuple[float, float],
) -> bool:
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def bbox_intersects_polygon(bbox: List[float], points: List[Dict[str, float]]) -> bool:
    x1, y1, x2, y2 = bbox
    rect = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    poly = [(p["x"], p["y"]) for p in points]

    if any(point_in_poly(corner, poly) for corner in rect):
        return True
    if any(x1 <= px <= x2 and y1 <= py <= y2 for px, py in poly):
        return True

    rect_edges = list(zip(rect, rect[1:] + rect[:1]))
    poly_edges = list(zip(poly, poly[1:] + poly[:1]))
    return any(segments_intersect(a, b, c, d) for a, b in rect_edges for c, d in poly_edges)


def bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def nms_detections(detections: List[Dict[str, object]]) -> List[Dict[str, object]]:
    kept: List[Dict[str, object]] = []
    for det in sorted(detections, key=lambda item: float(item["confidence"]), reverse=True):
        bbox = det["bbox"]
        if any(bbox_iou(bbox, kept_det["bbox"]) > IOU_THRESHOLD for kept_det in kept):
            continue
        kept.append(det)
        if len(kept) >= MAX_DETECTIONS:
            break
    return kept
