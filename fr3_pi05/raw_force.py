"""Raw arm stream for the bc2f_force checkpoints: a background poller and the window each observation carries.

The pi05_bc2f_* models read the flange wrench averaged over the 64 ms before each observation, computed from every
arm sample in that window (fr3_train/force_features.py). Training took those samples from the recorder's poller
(fr3_demo.recording.DemoCollector._arm_loop: its own Bamboo client, request/reply as fast as Bamboo answers,
~860 Hz); this module runs the same poller during a rollout and hands the runner the samples around an
observation's controller time.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable

import numpy as np

RAW_FORCE_COLUMNS = 15  # [bamboo time_sec, q (7), tau_J (7)]
RAW_FORCE_SEND_S = 0.128  # twice the model's 64 ms window, so the server never has to extrapolate
MIN_RATE_HZ = 500.0  # training data: 861 Hz mean, 85 % of gaps exactly 1 ms


class RawForceSampler:
    """Poll Bamboo in a background thread and keep the last `keep_s` seconds of (time_sec, q, tau_J).

    `client_factory` returns an object with ``get_joint_states() -> dict`` holding ``qpos``, ``tau_J`` and
    ``time_sec``, normally ``BambooFrankaClient(server_ip=..., control_port=..., enable_gripper=False)``: a client
    of its own, as the recorder used, so polling never waits on the control loop's requests. Repeated controller
    states (the same ``time_sec`` read twice) are kept once.
    """

    def __init__(self, client_factory: Callable[[], Any], rate_hz: float = 1000.0, keep_s: float = 2.0) -> None:
        if rate_hz <= 0 or keep_s <= RAW_FORCE_SEND_S:
            raise ValueError("rate_hz must be positive and keep_s longer than the sent window")
        self._factory = client_factory
        self._period = 1.0 / rate_hz
        self._samples: deque[np.ndarray] = deque(maxlen=int(keep_s * max(rate_hz, 1000.0)) + 1)
        self._lock = threading.Lock()
        self._new_sample = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._last_host_time = 0.0
        self._thread = threading.Thread(target=self._loop, name="bc2f-raw-force", daemon=True)

    def start(self) -> RawForceSampler:
        self._thread.start()
        return self

    def _loop(self) -> None:
        client = None
        try:
            client = self._factory()
            while not self._stop.is_set():
                started = time.monotonic()
                state = client.get_joint_states()
                if "tau_J" not in state or "time_sec" not in state:
                    raise RuntimeError("Robot state has no tau_J/time_sec; this Bamboo controller cannot serve bc2f_force")
                row = np.empty(RAW_FORCE_COLUMNS, dtype=np.float64)
                row[0] = float(state["time_sec"])
                row[1:8] = np.asarray(state["qpos"], dtype=np.float64)
                row[8:15] = np.asarray(state["tau_J"], dtype=np.float64)
                if not np.all(np.isfinite(row)):
                    raise RuntimeError("Bamboo returned a non-finite joint state")
                with self._new_sample:
                    if not self._samples or row[0] > self._samples[-1][0]:
                        self._samples.append(row)
                        self._last_host_time = time.monotonic()
                        self._new_sample.notify_all()
                self._ready.set()
                remaining = self._period - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
        except BaseException as error:  # noqa: BLE001 - surfaced to the control thread by check()
            self._error = error
            with self._new_sample:
                self._new_sample.notify_all()
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - shutdown must continue
                    pass

    def wait_ready(self, timeout: float = 5.0, min_span_s: float = 0.5) -> None:
        """Block until the buffer spans `min_span_s` and the achieved rate is acceptable."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.check()
            with self._lock:
                span = self._samples[-1][0] - self._samples[0][0] if len(self._samples) > 1 else 0.0
            if span >= min_span_s:
                rate = self.stats()["rate_hz"]
                if rate < MIN_RATE_HZ:
                    raise RuntimeError(f"Raw force poller reaches only {rate:.0f} Hz (< {MIN_RATE_HZ:.0f} Hz)")
                return
            time.sleep(0.01)
        raise RuntimeError(f"Raw force poller did not collect {min_span_s:.1f} s of samples within {timeout:.1f} s")

    def check(self, max_age_s: float = 0.1) -> None:
        if self._error is not None:
            raise RuntimeError(f"Raw force poller failed: {self._error!r}") from self._error
        if self._ready.is_set() and time.monotonic() - self._last_host_time > max_age_s:
            raise RuntimeError(f"Raw force poller stalled: no new arm sample for {max_age_s * 1000:.0f} ms")

    def window(self, robot_timestamp: float, span_s: float = RAW_FORCE_SEND_S, wait_s: float = 0.02) -> np.ndarray:
        """Samples with time_sec > robot_timestamp - span_s (newer ones included), as [N, 15] float64.

        Waits up to `wait_s` for the poller to pass `robot_timestamp`, so the window is complete up to the
        observation's own instant even if the poller was briefly descheduled."""
        deadline = time.monotonic() + wait_s
        with self._new_sample:
            while (not self._samples or self._samples[-1][0] < robot_timestamp) and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._new_sample.wait(remaining)
            rows = [row for row in self._samples if row[0] > robot_timestamp - span_s]
        self.check()
        if len(rows) < 2:
            raise RuntimeError(f"Raw force window around t={robot_timestamp:.3f} holds {len(rows)} samples")
        return np.stack(rows)

    def stats(self, last_s: float = 1.0) -> dict[str, float]:
        with self._lock:
            t = np.array([row[0] for row in self._samples])
        if len(t) < 2:
            return {"rate_hz": 0.0, "max_gap_ms": float("inf"), "samples": float(len(t))}
        t = t[t >= t[-1] - last_s]
        gaps = np.diff(t)
        return {"rate_hz": float((len(t) - 1) / max(t[-1] - t[0], 1e-9)),
                "max_gap_ms": float(gaps.max() * 1000) if len(gaps) else float("inf"), "samples": float(len(t))}

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
