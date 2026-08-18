"""Linux joystick reader with no SDL/ROS dependency."""

from __future__ import annotations

import array
import fcntl
import os
import select
import struct
import threading
import time
from dataclasses import dataclass

_EVENT = struct.Struct("<IhBB")
_EVENT_BUTTON = 0x01
_EVENT_AXIS = 0x02
_EVENT_INIT = 0x80


@dataclass(frozen=True)
class JoystickSnapshot:
    axes: tuple[float, ...]
    buttons: tuple[bool, ...]
    press_counts: tuple[int, ...]
    connected: bool
    name: str
    timestamp: float


class LinuxJoystick:
    """Continuously drain a Linux ``/dev/input/js*`` device in a thread."""

    def __init__(self, path: str = "/dev/input/js0") -> None:
        self.path = path
        self._fd: int | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._axes: list[float] = []
        self._buttons: list[bool] = []
        self._press_counts: list[int] = []
        self._connected = False
        self._name = "unknown"
        self._timestamp = 0.0

    @staticmethod
    def _ioctl_byte(fd: int, request: int) -> int:
        value = array.array("B", [0])
        fcntl.ioctl(fd, request, value)
        return int(value[0])

    @staticmethod
    def _ioctl_name(fd: int) -> str:
        value = array.array("B", [0] * 128)
        fcntl.ioctl(fd, 0x80806A13, value)  # JSIOCGNAME(128)
        return bytes(value).split(b"\0", 1)[0].decode(errors="replace")

    def open(self) -> LinuxJoystick:
        if self._fd is not None:
            return self
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as error:
            raise RuntimeError(f"Cannot open joystick {self.path}: {error}") from error

        try:
            axis_count = self._ioctl_byte(fd, 0x80016A11)  # JSIOCGAXES
            button_count = self._ioctl_byte(fd, 0x80016A12)  # JSIOCGBUTTONS
            name = self._ioctl_name(fd)
        except Exception:
            os.close(fd)
            raise

        with self._lock:
            self._fd = fd
            self._axes = [0.0] * axis_count
            self._buttons = [False] * button_count
            self._press_counts = [0] * button_count
            self._connected = True
            self._name = name
            self._timestamp = time.monotonic()

        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, name="joystick-reader", daemon=True)
        self._thread.start()
        return self

    def _read_loop(self) -> None:
        assert self._fd is not None
        poller = select.poll()
        poller.register(self._fd, select.POLLIN | select.POLLERR | select.POLLHUP)
        try:
            while not self._stop.is_set():
                events = poller.poll(100)
                for _, flags in events:
                    if flags & (select.POLLERR | select.POLLHUP):
                        return
                    while True:
                        try:
                            payload = os.read(self._fd, _EVENT.size)
                        except BlockingIOError:
                            break
                        except OSError:
                            return
                        if len(payload) != _EVENT.size:
                            return
                        _, value, event_type, number = _EVENT.unpack(payload)
                        event_type &= ~_EVENT_INIT
                        now = time.monotonic()
                        with self._lock:
                            if event_type == _EVENT_AXIS and number < len(self._axes):
                                self._axes[number] = max(-1.0, min(1.0, value / 32767.0))
                            elif event_type == _EVENT_BUTTON and number < len(self._buttons):
                                pressed = bool(value)
                                if pressed and not self._buttons[number]:
                                    self._press_counts[number] += 1
                                self._buttons[number] = pressed
                            self._timestamp = now
        finally:
            with self._lock:
                self._connected = False

    def snapshot(self) -> JoystickSnapshot:
        with self._lock:
            return JoystickSnapshot(
                axes=tuple(self._axes),
                buttons=tuple(self._buttons),
                press_counts=tuple(self._press_counts),
                connected=self._connected,
                name=self._name,
                timestamp=self._timestamp,
            )

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        fd = self._fd
        self._fd = None
        if fd is not None:
            os.close(fd)

    def __enter__(self) -> LinuxJoystick:  # noqa: PYI034
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()


def shaped_axis(value: float, deadzone: float = 0.12, exponent: float = 1.5) -> float:
    """Remove a deadzone, rescale the remaining range, and apply an expo curve."""

    magnitude = abs(float(value))
    if magnitude <= deadzone:
        return 0.0
    scaled = min(1.0, (magnitude - deadzone) / (1.0 - deadzone))
    return (-1.0 if value < 0 else 1.0) * scaled**exponent


def axis(snapshot: JoystickSnapshot, index: int) -> float:
    return snapshot.axes[index] if 0 <= index < len(snapshot.axes) else 0.0


def button(snapshot: JoystickSnapshot, index: int) -> bool:
    return snapshot.buttons[index] if 0 <= index < len(snapshot.buttons) else False
