# Flask YOLO Human Zone Detector

Small Flask app for Raspberry Pi 5 1GB RAM. It streams an RTSP camera, detects humans with YOLO, lets you drag 4 dots into a quadrangle, and raises repeated web alerts while a person bounding box overlaps the zone.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Export YOLOv8n to NCNN on a machine with enough RAM, then copy the output folder to `models/yolov8n_ncnn_model`:

```bash
yolo export model=yolov8n.pt format=ncnn imgsz=320
```

## Run

```bash
python app.py
```

Open `http://<raspberry-pi-ip>:5000`.

Edit `.env` before running.

IP camera `.env` values:

```bash
MODE=IPCAM
RTSP_URL=rtsp://user:pass@192.168.2.64:554/Streaming/Channels/102
```

Raspberry Pi camera `.env` value:

```bash
MODE=RPICAM
```

## Config

Settings live in `.env`:

- `MODE`: camera mode, `IPCAM` or `RPICAM`, default `IPCAM`.
- `RTSP_URL`: full RTSP URL. If unset, app builds one from camera fields below.
- `CAMERA_IP`: camera IP, default `192.168.2.64`.
- `CAMERA_USERNAME`: camera username, default `admin`.
- `CAMERA_PASSWORD`: camera password, default empty.
- `CAMERA_CHANNEL`: camera channel, default `102` for lower-latency substream.
- `RPICAM_COMMAND`: Raspberry Pi camera command, default `rpicam-vid`.
- `RPICAM_EXTRA_ARGS`: extra args appended to `rpicam-vid`.
- `YOLO_MODEL`: model path, default `models/yolov8n_ncnn_model`.
- `INFER_SIZE`: YOLO inference size, default `320`.
- `CONFIDENCE`: person confidence threshold, default `0.45`.
- `IOU_THRESHOLD`: YOLO duplicate-box merge threshold, default `0.45`.
- `MAX_DETECTIONS`: max person boxes per frame, default `5`.
- `MIN_PERSON_AREA_RATIO`: drop boxes smaller than this frame area ratio, default `0.0005`; set `-1` to disable this layer.
- `MIN_PERSON_HEIGHT_RATIO`: drop boxes shorter than this frame height ratio, default `0.025`; set `-1` to disable this layer.
- `MIN_PERSON_ASPECT`: drop person boxes narrower than this width/height ratio, default `0.08`; set `-1` to disable this layer.
- `MAX_PERSON_ASPECT`: drop person boxes wider than this width/height ratio, default `2.2`; set `-1` to disable this layer.
- `KEEP_BEST_PERSON_ONLY`: keep only strongest person box, default `1`; set `-1` to disable this layer.
- `CAMERA_WIDTH`: capture width, default `640`.
- `CAMERA_HEIGHT`: capture height, default `480`.
- `CAMERA_FPS`: capture FPS target for `rpicam-vid`, default `15`.
- `JPEG_QUALITY`: MJPEG quality, default `75`.
- `ALERT_INTERVAL`: seconds between repeated alerts while occupied, default `1.0`.
- `HID_ENABLED`: set `0` to disable USB HID notify, default `1`.
- `HID_DEVICE`: Linux hidraw path, default `/dev/hidraw0`.
- `PORT`: Flask port, default `5000`.

Zone points save to `zone_config.json`.

## Project layout

- `app.py`: Flask routes and MJPEG response.
- `config.py`: environment settings and camera/model defaults.
- `services/video_state.py`: camera threads, frame processing, snapshots, runtime filter toggles.
- `services/detector.py`: YOLO loading and person filter layers.
- `services/geometry.py`: zone intersection and detection NMS helpers.
- `services/notifier.py`: USB HID on/off output.
- `services/zone_store.py`: zone default/load/save helpers.

## Behavior

- Only COCO class `person` is detected.
- Detection filter layers can be toggled live in the web UI: `MIN_PERSON_HEIGHT_RATIO`, `MIN_PERSON_AREA_RATIO`, `MIN_PERSON_ASPECT`, `MAX_PERSON_ASPECT`, and `KEEP_BEST_PERSON_ONLY`.
- Collision means any person bounding box intersects the quadrangle.
- Capture thread always drains RTSP and detector processes latest frame only, reducing buffer delay.
- Alert repeats every `ALERT_INTERVAL` while a detected person overlaps the zone.
- Alert stops after no detected person overlaps the zone.
- USB HID writes direct hidraw packets: `ON\n\r\x00\x00\x00` when occupied starts and `OFF\n\r\x00\x00` when occupied clears.

RPICAM mode uses `rpicam-vid --codec mjpeg --output -` and parses JPEG frames from stdout. It does not use OpenCV GStreamer camera input and does not use H264/H265, because Raspberry Pi 5 has no H264/H265 hardware encode/decode.
