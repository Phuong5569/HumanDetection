from typing import Dict, Optional

from config import HID_DEVICE, HID_ENABLED


class HidNotifier:
    def __init__(self) -> None:
        self.enabled = HID_ENABLED
        self.error: Optional[str] = None
        self.last_state: Optional[bool] = None

    def send_state(self, occupied: bool) -> None:
        if not self.enabled or occupied == self.last_state:
            return

        payload = b"ON\n\r" + b"\x00" * 3 if occupied else b"OFF\n\r" + b"\x00" * 2

        try:
            with open(HID_DEVICE, "wb", buffering=0) as hidraw:
                hidraw.write(payload)
            self.last_state = occupied
            self.error = None
        except Exception as exc:
            self.error = str(exc)

    def snapshot(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "device": HID_DEVICE,
            "last_state": self.last_state,
            "error": self.error,
        }
