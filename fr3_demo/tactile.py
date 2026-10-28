"""GelSight Mini tactile capture for synchronized demonstration recording.

The sensor is a UVC camera looking at a deformable gel pad. It exposes exactly
one mode, 3280x2464 MJPEG at 25 fps, so every frame is an 8 MP JPEG decode and
25 Hz is a hardware ceiling rather than a choice. Frames are downscaled on the
capture thread: the gel image carries nothing like 8 MP of information, and
keeping the native size would cost far more disk and CPU than it returns.

Capture runs at the sensor's full rate on its own timeline, independent of the
15 Hz recording grid, the same way raw_force does for arm torque.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# One UVC mode, fixed in hardware.
NATIVE_SIZE = (3280, 2464)
NATIVE_FPS = 25.0
DEVICE_NAME = "GelSight"
VIDEO_CLASS_ROOT = Path("/sys/class/video4linux")


@dataclass(frozen=True)
class TactileFrame:
    """One captured gel image, already downscaled, on the host monotonic clock."""

    image: np.ndarray
    captured_monotonic: float
    frame_number: int

    @property
    def alignment_timestamp(self) -> float:
        return self.captured_monotonic


def _video_nodes() -> list[tuple[int, str]]:
    nodes = []
    if not VIDEO_CLASS_ROOT.is_dir():
        return nodes
    for entry in sorted(VIDEO_CLASS_ROOT.iterdir()):
        if not entry.name.startswith("video"):
            continue
        try:
            # A UVC device publishes several nodes; only index 0 captures video,
            # the others carry metadata and cannot be read as images.
            if (entry / "index").read_text().strip() != "0":
                continue
            name = (entry / "name").read_text().strip()
        except OSError:
            continue
        nodes.append((int(entry.name.removeprefix("video")), name))
    return nodes


def find_tactile_device(name_contains: str = DEVICE_NAME) -> int:
    """Return the V4L2 index of the tactile sensor's capture node."""

    nodes = _video_nodes()
    wanted = name_contains.lower()
    for index, name in nodes:
        if wanted in name.lower():
            return index
    available = ", ".join(f"video{index} ({name})" for index, name in nodes) or "none"
    raise RuntimeError(
        f"No video capture device whose name contains {name_contains!r} was found. Available: {available}"
    )


def resolve_device(device: str | int | None) -> int:
    """Accept an index, a /dev/videoN path, or None to auto-detect."""

    if device is None or device == "":
        return find_tactile_device()
    if isinstance(device, int):
        return device
    text = str(device).strip()
    if text.isdigit():
        return int(text)
    name = Path(text).name
    if name.startswith("video") and name.removeprefix("video").isdigit():
        return int(name.removeprefix("video"))
    raise RuntimeError(f"Cannot interpret tactile device {device!r}; use an index or /dev/videoN")


class GelSightCamera:
    """Full-rate tactile capture with a bounded buffer for grid alignment.

    `on_frame` is called on the capture thread for every frame, which is how a
    recording keeps the complete stream; the internal buffer only serves
    nearest-frame alignment and preview, and deliberately holds a short window
    so a stalled consumer cannot grow it without bound.
    """

    def __init__(
        self,
        device: str | int | None = None,
        width: int = 320,
        height: int = 240,
        fps: float = NATIVE_FPS,
        rotate_180: bool = False,
        on_frame: Callable[[TactileFrame], None] | None = None,
    ) -> None:
        self.device_index = resolve_device(device)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(min(fps, NATIVE_FPS))
        self.rotate_180 = bool(rotate_180)
        self._on_frame = on_frame
        self._frames: list[TactileFrame] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._capture: Any = None
        self._frame_number = 0
        self._dropped = 0
        # Two seconds of history is plenty for a 15 Hz grid and bounds memory at
        # 50 frames of 320x240 RGB, about 11 MB.
        self._buffer_size = max(8, int(self.fps * 2))

    @staticmethod
    def _open(index: int) -> Any:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("Tactile recording requires: pip install -e '.[tactile]'") from error
        capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Cannot open tactile sensor at /dev/video{index}. A UVC device has one owner, so "
                f"close any viewer holding it (check with: fuser -v /dev/video{index})."
            )
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # The frames we skip are decoded-and-discarded work; keep the driver
        # queue short so a grab returns the newest gel state, not a backlog.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Hand us the raw MJPEG buffer instead of a decoded 8 MP BGR frame, so
        # the decode can be done at reduced scale. Measured on this rig: full
        # decode plus resize is 31.6 ms per frame, 79% of a core at 25 Hz, while
        # a 1/8-scale decode plus resize is 4.3 ms. Not every OpenCV build
        # honours this, so the capture loop handles both shapes.
        capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        return capture

    def _reduced_flag(self) -> int:
        """Largest libjpeg reduction that still exceeds the stored size.

        Reducing in the DCT domain is far cheaper than decoding in full and
        resizing, but going below the target would mean upscaling, so the
        factor is chosen from the configured size rather than fixed.
        """

        import cv2

        for factor, flag in (
            (8, cv2.IMREAD_REDUCED_COLOR_8),
            (4, cv2.IMREAD_REDUCED_COLOR_4),
            (2, cv2.IMREAD_REDUCED_COLOR_2),
        ):
            if NATIVE_SIZE[0] // factor >= self.width and NATIVE_SIZE[1] // factor >= self.height:
                return flag
        return cv2.IMREAD_COLOR

    def set_frame_observer(self, observer: Callable[[TactileFrame], None] | None) -> None:
        self._on_frame = observer

    def start(self, timeout: float = 10.0) -> None:
        if self._thread is not None:
            raise RuntimeError("Tactile capture is already running")
        self._thread = threading.Thread(target=self._loop, name="tactile-capture", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            self.close()
            raise RuntimeError(f"Tactile sensor produced no frame within {timeout:.0f} s")
        self.check_health()

    def _downscale(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        # INTER_AREA is the correct filter for a large reduction; anything else
        # aliases the gel's printed marker dots.
        if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _loop(self) -> None:
        try:
            self._capture = self._open(self.device_index)
            reduced = self._reduced_flag()
            period = 1.0 / self.fps if self.fps > 0 else 0.0
            while not self._stop.is_set():
                started = time.monotonic()
                ok = self._capture.grab()
                if not ok:
                    raise RuntimeError("Tactile sensor stopped delivering frames")
                ok, raw = self._capture.retrieve()
                if ok and raw is not None and raw.ndim <= 2:
                    # Raw MJPEG payload: decode it ourselves, at reduced scale.
                    import cv2

                    raw = cv2.imdecode(raw.reshape(-1), reduced)
                received = time.monotonic()
                if not ok or raw is None:
                    self._dropped += 1
                    continue
                image = self._downscale(raw)
                # Midpoint of the grab window, matching how the arm sampler
                # timestamps a request/reply exchange.
                frame = TactileFrame(
                    image=image,
                    captured_monotonic=(started + received) * 0.5,
                    frame_number=self._frame_number,
                )
                self._frame_number += 1
                with self._lock:
                    self._frames.append(frame)
                    if len(self._frames) > self._buffer_size:
                        del self._frames[: len(self._frames) - self._buffer_size]
                observer = self._on_frame
                if observer is not None:
                    observer(frame)
                self._ready.set()
                remaining = period - (time.monotonic() - started)
                if remaining > 0:
                    self._stop.wait(remaining)
        except BaseException as error:  # noqa: BLE001 - surfaced through check_health
            if not self._stop.is_set():
                self._error = error
            self._ready.set()
        finally:
            capture, self._capture = self._capture, None
            if capture is not None:
                capture.release()

    def check_health(self) -> None:
        error = self._error
        if error is not None:
            raise RuntimeError(f"Tactile capture failed: {error}") from error

    @property
    def frames_captured(self) -> int:
        return self._frame_number

    @property
    def frames_dropped(self) -> int:
        return self._dropped

    def latest(self, max_age: float = 0.5) -> TactileFrame | None:
        self.check_health()
        with self._lock:
            frame = self._frames[-1] if self._frames else None
        if frame is None:
            return None
        if max_age > 0 and (time.monotonic() - frame.captured_monotonic) > max_age:
            return None
        return frame

    def nearest(self, target: float, max_delta: float) -> TactileFrame:
        """The buffered frame closest to a grid instant, as the cameras do."""

        self.check_health()
        with self._lock:
            frames = list(self._frames)
        if not frames:
            raise RuntimeError("Tactile sensor has produced no frames yet")
        best = min(frames, key=lambda frame: abs(frame.alignment_timestamp - target))
        delta = abs(best.alignment_timestamp - target)
        if delta > max_delta:
            raise RuntimeError(
                f"Tactile sensor cannot align target: nearest frame is {delta * 1000.0:.1f} ms away"
            )
        return best

    def close(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5.0)

    def __enter__(self) -> GelSightCamera:  # noqa: PYI034 - typing.Self needs 3.11, this supports 3.10
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def measure_rate(
    device: str | int | None = None,
    seconds: float = 10.0,
    width: int = 320,
    height: int = 240,
) -> dict[str, float]:
    """Measure what the sensor actually delivers, for reporting and config checks."""

    stamps: list[float] = []
    camera = GelSightCamera(device=device, width=width, height=height, fps=NATIVE_FPS)
    camera.set_frame_observer(lambda frame: stamps.append(frame.captured_monotonic))
    camera.start()
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(0.05)
            camera.check_health()
    finally:
        camera.close()
    if len(stamps) < 3:
        raise RuntimeError(f"Tactile sensor produced only {len(stamps)} frames in {seconds:.0f} s")
    deltas = np.diff(np.asarray(stamps, dtype=np.float64)) * 1000.0
    span = stamps[-1] - stamps[0]
    return {
        "frames": float(len(stamps)),
        "seconds": float(span),
        "rate_hz": float((len(stamps) - 1) / span) if span > 0 else 0.0,
        "dt_p50_ms": float(np.median(deltas)),
        "dt_p95_ms": float(np.percentile(deltas, 95)),
        "dt_max_ms": float(deltas.max()),
        "dropped": float(camera.frames_dropped),
    }


def main(argv: list[str] | None = None) -> int:
    """Report what the tactile sensor delivers, without recording anything."""

    import argparse

    parser = argparse.ArgumentParser(description="Check the GelSight tactile sensor and measure its rate.")
    parser.add_argument("--device", default="", help="index or /dev/videoN; empty auto-detects")
    parser.add_argument("--seconds", type=float, default=10.0, help="measurement window")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    args = parser.parse_args(argv)

    for index, name in _video_nodes():
        marker = " <-- tactile" if DEVICE_NAME.lower() in name.lower() else ""
        print(f"  /dev/video{index}  {name}{marker}")
    device = resolve_device(args.device or None)
    print(f"\nMeasuring /dev/video{device} for {args.seconds:.0f} s at {args.width}x{args.height} ...")
    stats = measure_rate(device, seconds=args.seconds, width=args.width, height=args.height)
    print(
        f"  {stats['frames']:.0f} frames in {stats['seconds']:.1f} s -> {stats['rate_hz']:.2f} Hz\n"
        f"  frame interval p50 {stats['dt_p50_ms']:.1f} ms, p95 {stats['dt_p95_ms']:.1f} ms, "
        f"max {stats['dt_max_ms']:.1f} ms\n"
        f"  retrieve failures: {stats['dropped']:.0f}\n"
        f"  hardware mode: {NATIVE_SIZE[0]}x{NATIVE_SIZE[1]} MJPEG at {NATIVE_FPS:.0f} fps (the only mode)"
    )
    if stats["rate_hz"] < NATIVE_FPS * 0.8:
        print(
            f"\n  Note: {stats['rate_hz']:.1f} Hz is well under the {NATIVE_FPS:.0f} fps the sensor offers. "
            "Each frame is an 8 MP MJPEG decode, so this is usually CPU, not the sensor."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
