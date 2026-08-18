"""Asynchronous force feedback for the BETOP controller."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

LOG = logging.getLogger("fr3_teleop.feedback")


def _load_evdev() -> tuple[Any, Any, Any]:
    try:
        from evdev import InputDevice, ecodes, ff
    except ImportError as error:
        raise RuntimeError("Joystick vibration requires the recording extra: pip install -e '.[recording]'") from error
    return InputDevice, ecodes, ff


def find_feedback_device(name_contains: str = "BETOP") -> str:
    InputDevice, ecodes, _ = _load_evdev()
    for path in sorted(Path("/dev/input").glob("event*")):
        try:
            device = InputDevice(str(path))
            supports_rumble = ecodes.FF_RUMBLE in device.capabilities().get(ecodes.EV_FF, [])
            if name_contains.casefold() in device.name.casefold() and supports_rumble:
                return str(path)
        except (OSError, PermissionError):
            continue
    raise RuntimeError(f"No force-feedback input device containing {name_contains!r} was found")


class Rumbler:
    """Queue short rumble patterns without blocking the control loop."""

    def __init__(self, event_path: str | None = None) -> None:
        self.event_path = event_path or find_feedback_device()
        self._lock = threading.Lock()

    def _play(self, pulses: int, duration: float, gap: float) -> None:
        InputDevice, ecodes, ff = _load_evdev()
        with self._lock:
            device = InputDevice(self.event_path)
            for index in range(pulses):
                effect = ff.Effect(
                    ecodes.FF_RUMBLE,
                    -1,
                    0,
                    ff.Trigger(0, 0),
                    ff.Replay(int(duration * 1000), 0),
                    ff.EffectType(ff_rumble_effect=ff.Rumble(strong_magnitude=0x7000, weak_magnitude=0xA000)),
                )
                effect_id = device.upload_effect(effect)
                try:
                    device.write(ecodes.EV_FF, effect_id, 1)
                    time.sleep(duration)
                finally:
                    device.erase_effect(effect_id)
                if index + 1 < pulses:
                    time.sleep(gap)

    def play(self, pulses: int, duration: float = 0.14, gap: float = 0.10) -> None:
        def run() -> None:
            try:
                self._play(pulses, duration, gap)
            except Exception as error:  # noqa: BLE001 - feedback failure must never stop robot control
                LOG.warning("Joystick vibration failed: %s", error)

        threading.Thread(target=run, name="joystick-rumble", daemon=True).start()

    def recording_started(self) -> None:
        self.play(1, duration=0.20)

    def recording_stopped(self) -> None:
        self.play(2)

    def error(self) -> None:
        self.play(3, duration=0.08, gap=0.06)
