import time

from flask import Flask, Response, jsonify, render_template, request

from config import PORT
from services.video_state import VideoState
from services.zone_store import default_zone

app = Flask(__name__)
state = VideoState()


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/video")
def video() -> Response:
    def generate():
        while True:
            with state.lock:
                frame = state.encoded_frame
            if frame is None:
                time.sleep(0.1)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.01)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status() -> Response:
    return jsonify(state.snapshot())


@app.route("/api/filter-layers", methods=["GET", "POST"])
def api_filter_layers() -> Response:
    if request.method == "GET":
        return jsonify({"filter_layers": state.snapshot()["filter_layers"]})

    payload = request.get_json(silent=True) or {}
    updated = state.update_filter_layers(payload.get("filter_layers", {}))
    return jsonify({"ok": True, "filter_layers": updated})


@app.route("/api/zone", methods=["GET", "POST"])
def api_zone() -> Response:
    if request.method == "GET":
        return jsonify({"points": state.snapshot()["zone"]})

    payload = request.get_json(silent=True) or {}
    points = payload.get("points", [])
    if len(points) != 4:
        return jsonify({"error": "expected exactly 4 points"}), 400

    try:
        normalized = [{"x": float(p["x"]), "y": float(p["y"])} for p in points]
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "points must contain numeric x and y"}), 400

    state.update_zone(normalized)
    return jsonify({"ok": True, "points": normalized})


@app.route("/api/zone/reset", methods=["POST"])
def api_zone_reset() -> Response:
    points = default_zone(state.frame_size["width"], state.frame_size["height"])
    state.update_zone(points)
    return jsonify({"ok": True, "points": points})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
