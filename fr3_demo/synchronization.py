"""Clock mapping and timestamp helpers for synchronized data collection."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClockEstimate:
    """Current affine mapping from a sensor clock to host monotonic time."""

    scale: float
    offset: float
    sample_count: int
    residual_p95: float


class AffineClockMapper:
    """Estimate ``host_time = scale * source_time + offset`` online.

    Sensor clocks have an arbitrary epoch and drift slightly relative to the
    workstation monotonic clock.  A single offset therefore is not sufficient
    for long episodes.  Camera samples use a low-delay envelope because their
    host timestamp is taken after USB transport.  Request/reply sensors use the
    lowest round-trip-time samples and a median offset.
    """

    def __init__(
        self,
        *,
        max_samples: int = 600,
        lower_envelope: bool = False,
        fit_scale: bool = True,
        reset_on_discontinuity: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        self._samples: deque[tuple[float, float, float]] = deque(maxlen=max_samples)
        self._lower_envelope = lower_envelope
        self._fit_scale = fit_scale
        self._reset_on_discontinuity = reset_on_discontinuity
        self._last_source: float | None = None
        self._last_host: float | None = None

    def add(self, source_time: float, host_time: float, uncertainty: float = 0.0) -> None:
        if not np.isfinite(source_time) or not np.isfinite(host_time):
            return
        with self._lock:
            if self._reset_on_discontinuity and self._last_source is not None and self._last_host is not None:
                source_delta = source_time - self._last_source
                host_delta = host_time - self._last_host
                if source_delta < -1e-6 or abs(source_delta - host_delta) > 1.0:
                    # Device/controller restarted or changed timestamp domain.
                    self._samples.clear()
            self._last_source = source_time
            self._last_host = host_time
            self._samples.append((float(source_time), float(host_time), max(0.0, float(uncertainty))))

    def estimate(self) -> ClockEstimate:
        with self._lock:
            if not self._samples:
                raise RuntimeError("Clock mapper has no samples")
            samples = np.asarray(self._samples, dtype=np.float64)
        source = samples[:, 0]
        host = samples[:, 1]
        uncertainty = samples[:, 2]

        # Prefer the least-delayed request/reply observations.  Camera samples
        # all have uncertainty=0 and therefore all remain eligible.
        cutoff = float(np.percentile(uncertainty, 25.0))
        eligible = uncertainty <= cutoff + 1e-12
        if int(np.count_nonzero(eligible)) < min(8, len(samples)):
            eligible = np.ones(len(samples), dtype=bool)
        fit_source = source[eligible]
        fit_host = host[eligible]

        scale = 1.0
        centered = fit_source - float(np.mean(fit_source))
        denominator = float(centered @ centered)
        if self._fit_scale and len(fit_source) >= 8 and float(np.ptp(fit_source)) >= 0.5 and denominator > 1e-12:
            scale = float(centered @ (fit_host - float(np.mean(fit_host))) / denominator)
            # Real clocks drift in ppm, not percent.  This wider guard still
            # tolerates imperfect early observations without accepting a bad fit.
            scale = float(np.clip(scale, 0.995, 1.005))

        residual = fit_host - scale * fit_source
        percentile = 5.0 if self._lower_envelope else 50.0
        offset = float(np.percentile(residual, percentile))
        all_residual = np.abs(host - (scale * source + offset))
        return ClockEstimate(
            scale=scale,
            offset=offset,
            sample_count=len(samples),
            residual_p95=float(np.percentile(all_residual, 95.0)),
        )

    def to_host(self, source_time: float) -> float:
        estimate = self.estimate()
        return estimate.scale * float(source_time) + estimate.offset
