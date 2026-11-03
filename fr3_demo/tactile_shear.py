"""Gel shear from the GelSight marker dots, for a controller that must fail closed.

The gel carries a printed grid of dark dots. Shear moves them together, twist
turns them and pressing spreads them a little, so their motion against a frame
of the unloaded gel measures how far the gel surface has moved, which is what
tears it.

Dots are found as dark blobs against a morphological closing of the image, a
local upper envelope that follows the LEDs' colour gradient and broad contact
shading, and located by contrast-weighted centroids. Each reference dot is then
followed frame to frame: predicted from its last position plus the field's last
motion, and matched only to a detection that is the sole candidate inside a
gate well under half the dot spacing, on both sides. Following the dots, rather
than matching every frame against the reference, is what keeps their identity
once the accumulated shear passes half a spacing. A least-squares similarity
(translation, rotation, scale) fitted to the matched dots gives shear, twist
and spread.

Fail closed: whenever too few dots match or the fit does not explain them, the
estimate is valid=False with a reason and NaN in every motion field. Motion too
fast to follow safely, too few dots to carry identities, a poor fit or a long
gap between frames loses the track. So does a jump by about a whole spacing,
which lands each dot on its neighbour's spot and would otherwise read as a
small motion. The gel's displacement field is smooth, while a pairing one
spacing off carries the printed grid's jitter, so a field that turns smooth
when every dot is taken for its neighbour gives the jump away. A lost tracker
matches straight against the reference with a tight gate, which can only
succeed near the unloaded state, and accepts the result only if taking every
dot for its neighbour makes the field clearly rougher; otherwise a gel held
sheared by a whole spacing would pass for unloaded. Both checks need a printed
grid irregular enough to tell neighbours apart, and lens distortion does not
help: a gel shifted by a whole spacing images exactly like the unloaded gel. On
a perfectly regular grid a whole-spacing jump goes unseen, and a lost track
only comes back by re-zeroing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# Background brightness (0-255) below which a region counts as unlit: dividing
# by a near-black envelope would turn sensor noise into dark "dots".
_MIN_BACKGROUND = 40.0
# Neighbour distances on a dot grid vary by a few percent; between random dark
# blobs, by a third or more.
_MAX_SPACING_CV = 0.35
# A frame whose dark-blob mask covers more than this multiple of the
# reference's share, plus this share of the image, is noise or a corrupt
# decode, not the gel: its "dots" are not worth following.
_MAX_COVERAGE_RATIO = 3.0
_MAX_COVERAGE_EXCESS = 0.05
# Roughness below this fraction of the spacing is centroid noise, not evidence
# for one dot identity over another.
_ALIAS_NOISE_FLOOR = 0.005
_NAN_ROW = np.full((1, 2), np.nan)
# Equal channel weights: each LED colour lights its own side of the gel, so no
# single channel, and no luminance weighting, is bright everywhere.
_CHANNEL_MEAN = np.full((1, 3), 1.0 / 3.0, dtype=np.float32)
_LOST_ADVICE = "unload the gel or re-zero"
_AMBIGUOUS_ADVICE = "the markers also fit the reference shifted by one spacing, so re-zero with the gel unloaded"


@dataclass(frozen=True)
class ShearEstimate:
    """The marker field's motion against the reference, or why there is none.

    Image pixels as the frame arrives (after any rotate_180), x right and y
    down, with pixel centres at integer coordinates. ``shear_px`` is how far
    the centroid of the matched reference dots moved; ``twist_rad`` and
    ``spread`` (scale - 1) are about that centroid; ``max_displacement_px`` is
    the largest single dot's motion. ``twist_rad`` is positive for a
    counter-clockwise turn of the dots as the image is displayed (row 0 at the
    top), the usual math sense with y up; measured directly in the x-right,
    y-down pixel axes the same turn has the opposite sign. When ``valid`` is
    False every motion field is NaN, never a number to act on.
    """

    timestamp: float
    frame_number: int
    valid: bool
    reason: str
    shear_px: tuple[float, float]
    twist_rad: float
    spread: float
    max_displacement_px: float
    matched_fraction: float
    marker_spacing_px: float


@dataclass(frozen=True)
class MarkerParams:
    """Tunables. Gates, steps and residuals are fractions of the reference dot spacing."""

    # Fewest dots for a usable reference, and fewest followed to keep the track.
    min_markers: int = 12
    # Below this share of reference dots matched the estimate is invalid; the track continues.
    min_matched_fraction: float = 0.6
    # How much darker than its local background a pixel must be to belong to a dot.
    min_contrast: float = 0.2
    # Dot area bounds as fractions of the image area, used to find the reference dots.
    min_area_fraction: float = 4e-5
    max_area_fraction: float = 6e-3
    # Afterwards a dot's area must be within this factor of the reference median.
    area_ratio: float = 3.0
    # Roundness: axis ratio from second moments, and the share of the bounding box filled.
    max_elongation: float = 2.0
    min_fill: float = 0.45
    # Gate around a dot's predicted position when following it.
    track_gate: float = 0.35
    # Largest median dot motion between two updates; more loses the track.
    max_step: float = 0.25
    # Gate when matching straight against the reference while lost, and when a dot rejoins from the fit.
    reacquire_gate: float = 0.2
    # Largest RMS distance of the matched dots from the fitted similarity.
    max_residual: float = 0.2
    # Longest time between tracked frames, in seconds, before the track is lost.
    max_gap_s: float = 0.25
    # How decisively one pairing must beat the pairing one spacing over: to re-acquire, the shifted
    # one must be this many times rougher; to lose the track, this many times smoother.
    alias_ratio: float = 2.0

    def __post_init__(self) -> None:
        checks = (
            (self.min_markers >= 4, "min_markers must be at least 4"),
            (0.0 < self.min_matched_fraction <= 1.0, "min_matched_fraction must be in (0, 1]"),
            (0.0 < self.min_contrast < 1.0, "min_contrast must be in (0, 1)"),
            (
                0.0 < self.min_area_fraction < self.max_area_fraction < 1.0,
                "need 0 < min_area_fraction < max_area_fraction < 1",
            ),
            (self.area_ratio > 1.0, "area_ratio must exceed 1"),
            (self.max_elongation >= 1.0, "max_elongation must be at least 1"),
            (0.0 <= self.min_fill < 1.0, "min_fill must be in [0, 1)"),
            # At half a spacing the neighbouring dot is as close as the right one.
            (0.0 < self.track_gate < 0.5, "track_gate must be under half the marker spacing"),
            (0.0 < self.max_step < 0.5, "max_step must be under half the marker spacing"),
            (0.0 < self.reacquire_gate < 0.5, "reacquire_gate must be under half the marker spacing"),
            (self.max_residual > 0.0, "max_residual must be positive"),
            (self.max_gap_s > 0.0, "max_gap_s must be positive"),
            (self.alias_ratio >= 1.0, "alias_ratio must be at least 1"),
        )
        for ok, message in checks:
            if not ok:
                raise ValueError(message)


@dataclass(frozen=True)
class _BlobLimits:
    kernel: int  # background closing, px; must exceed a dot's diameter / sqrt(2)
    min_area: float  # px
    max_area: float  # px


@dataclass(frozen=True)
class _Fit:
    """Least-squares similarity p -> target + [[a, -b], [b, a]] (p - origin), in pixel axes.

    hypot(a, b) is the scale, and atan2(b, a) turns from +x towards +y, which
    is clockwise as the image is displayed.
    """

    origin: np.ndarray
    target: np.ndarray
    a: float
    b: float
    rms: float  # distance of the points from the fit, px

    def apply(self, points: np.ndarray) -> np.ndarray:
        offset = points - self.origin
        return self.target + np.stack(
            (self.a * offset[:, 0] - self.b * offset[:, 1], self.b * offset[:, 0] + self.a * offset[:, 1]), axis=1
        )


class _Lost(Exception):
    """The dots cannot be carried into this frame without risking a wrong identity."""


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Tactile shear estimation requires: pip install -e '.[tactile]'") from error
    return cv2


def _odd(value: float) -> int:
    return max(3, round(value) | 1)


def _image_problem(image: Any) -> str:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        shape = getattr(image, "shape", None)
        kind = getattr(image, "dtype", type(image).__name__)
        return f"expected an H x W x 3 uint8 RGB image, got {kind} with shape {shape}"
    if min(image.shape[:2]) < 32:
        return f"a {image.shape[1]}x{image.shape[0]} image is too small to find markers in"
    return ""


def _fit_similarity(reference: np.ndarray, current: np.ndarray) -> _Fit:
    origin = reference.mean(axis=0)
    target = current.mean(axis=0)
    p = reference - origin
    q = current - target
    norm = float(np.sum(p * p))
    if not norm > 0.0:
        return _Fit(origin, target, math.nan, math.nan, math.nan)
    px, py, qx, qy = p[:, 0], p[:, 1], q[:, 0], q[:, 1]
    a = float(np.dot(px, qx) + np.dot(py, qy)) / norm
    b = float(np.dot(px, qy) - np.dot(py, qx)) / norm
    ex = qx - (a * px - b * py)
    ey = qy - (b * px + a * py)
    return _Fit(origin, target, a, b, math.sqrt(float(np.mean(ex * ex + ey * ey))))


def _match(predicted: np.ndarray, detected: np.ndarray, gate: float) -> tuple[np.ndarray, np.ndarray]:
    """Pair a prediction with a detection when each is the other's only candidate inside the gate.

    A second candidate inside the gate, on either side, makes the pairing
    ambiguous, so it is dropped rather than guessed. Returns index arrays into
    ``predicted`` and ``detected``.
    """

    if len(predicted) == 0 or len(detected) == 0:
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty
    dx = predicted[:, 0, None] - detected[None, :, 0]
    dy = predicted[:, 1, None] - detected[None, :, 1]
    inside = dx * dx + dy * dy <= gate * gate
    rows = np.flatnonzero(np.count_nonzero(inside, axis=1) == 1)
    columns = inside[rows].argmax(axis=1)
    unique = np.count_nonzero(inside, axis=0)[columns] == 1
    return rows[unique], columns[unique]


def _median(values: np.ndarray, axis: int) -> np.ndarray:
    """Median along ``axis`` of the finite values, NaN where there are none, keeping the axis.

    np.nanmedian warns on an empty slice, and silencing warnings is not
    thread-safe; NaN sorts last, so the middle of the finite values is found
    by count.
    """

    count = np.count_nonzero(np.isfinite(values), axis=axis, keepdims=True)
    ordered = np.sort(values, axis=axis)
    low = np.take_along_axis(ordered, np.maximum(count - 1, 0) // 2, axis=axis)
    high = np.take_along_axis(ordered, count // 2, axis=axis)
    return 0.5 * (low + high)


def _second_differences(field: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Along each grid axis, each dot's vector minus the mean of its two neighbours' there.

    ``field`` is (..., dots, 2) and ``axes`` holds, per axis, the neighbour
    indices on either side (-1 for none). Returns (..., axes, dots, 2), NaN
    where a vector is unknown.
    """

    padded = np.concatenate((field, np.full((*field.shape[:-2], 1, 2), np.nan)), axis=-2)
    return field[..., None, :, :] - 0.5 * (padded[..., axes[:, 0], :] + padded[..., axes[:, 1], :])


def _roughness(differences: np.ndarray) -> np.ndarray:
    """Per grid step, the median size of the second differences once each axis's common part is removed.

    ``differences`` is (steps, axes, dots, 2), NaN for dots left out. A smooth
    field leaves second differences that are nearly the same everywhere (lens
    curvature, say); what varies from dot to dot is the printed grid's jitter
    and centroid noise.
    """

    centred = differences - _median(differences, axis=2)
    sizes = np.hypot(centred[..., 0], centred[..., 1])
    return _median(sizes.reshape(len(sizes), -1), axis=1)[:, 0]


def _pairing_roughness(
    source: np.ndarray, current: np.ndarray, neighbours: np.ndarray, axes: np.ndarray, fewest: int
) -> list[tuple[float, float]]:
    """Roughness of the displacement field with the dots as paired, and with each one's neighbour's dot.

    ``source`` and ``current`` hold a position per reference dot, NaN where
    unknown. Moving every identity one grid step adds the grid's own jitter to
    a smooth field, or removes it if the pairing was already one step off. One
    (own, shifted) pair per grid step, both measured on the same dots, for each
    step that leaves at least ``fewest`` of them and half of the own field's.
    """

    own = _second_differences(current - source, axes)
    known = np.isfinite(own[..., 0])
    needed = max(fewest, 0.5 * np.count_nonzero(known))
    shifted = _second_differences(np.vstack((current, _NAN_ROW))[neighbours] - source, axes)
    both = known & np.isfinite(shifted[..., 0])
    own_rough = _roughness(np.where(both[..., None], own, np.nan))
    shifted_rough = _roughness(np.where(both[..., None], shifted, np.nan))
    counts = np.count_nonzero(both, axis=(1, 2))
    return [(float(o), float(s)) for o, s, n in zip(own_rough, shifted_rough, counts) if n >= needed]


def _neighbour_distances(points: np.ndarray) -> np.ndarray:
    offset = points[:, None, :] - points[None, :, :]
    distance = np.hypot(offset[..., 0], offset[..., 1])
    np.fill_diagonal(distance, np.inf)
    return distance.min(axis=1)


def _lattice_shifts(points: np.ndarray, spacing: float) -> np.ndarray:
    """Shifts that carry the dot grid nearly onto itself: the offsets most dots have to a neighbour.

    Taken around the dots nearest the middle, where lens distortion bends the
    grid least, and averaged over them. A shift counts only when most of those
    dots have a neighbour there; a grid gives four or more, random blobs none.
    """

    centre = np.median(points, axis=0)
    core = points[np.argsort(np.hypot(*(points - centre).T))[:40]]
    offsets = (points[None, :, :] - core[:, None, :]).reshape(-1, 2)
    length = np.hypot(offsets[:, 0], offsets[:, 1])
    offsets = offsets[(length > 0.5 * spacing) & (length < 1.6 * spacing)]
    gap = offsets[:, None, :] - offsets[None, :, :]
    close = gap[..., 0] ** 2 + gap[..., 1] ** 2 < (0.25 * spacing) ** 2
    alive = np.ones(len(offsets), dtype=bool)
    shifts = []
    while alive.any():
        support = (close & alive).sum(axis=1) * alive
        best = int(support.argmax())
        if support[best] < 0.5 * len(core):
            break
        members = close[best] & alive
        shifts.append(offsets[members].mean(axis=0))
        alive &= ~members
    return np.asarray(shifts, dtype=np.float64).reshape(-1, 2)


def _neighbour_table(points: np.ndarray, shifts: np.ndarray, spacing: float) -> np.ndarray:
    """For each grid shift, the index of each dot's neighbour there, or -1.

    Found dot by dot, so the table follows the grid where lens distortion
    bends it away from the shifts measured in the middle.
    """

    table = np.full((len(shifts), len(points)), -1, dtype=np.intp)
    for row, shift in enumerate(shifts):
        dots, neighbours = _match(points + shift, points, 0.3 * spacing)
        table[row, dots] = neighbours
    return table


def _grid_axes(shifts: np.ndarray) -> np.ndarray:
    """Pairs of opposite shifts among the shortest ones: the grid's axes, as (axes, 2) shift indices."""

    if len(shifts) == 0:
        return np.zeros((0, 2), dtype=np.intp)
    length = np.hypot(shifts[:, 0], shifts[:, 1])
    shortest = length.min()
    axes = []
    for row in np.flatnonzero(length < 1.25 * shortest):
        gap = np.hypot(*(shifts + shifts[row]).T)
        opposite = int(gap.argmin())
        if row < opposite and gap[opposite] < 0.25 * shortest:
            axes.append((row, opposite))
    return np.asarray(axes, dtype=np.intp).reshape(-1, 2)


class MarkerShearTracker:
    """Measure how far the gel surface has moved from its marker dots; see the module docstring.

    Not thread-safe: set_reference, update and detect share state, so a single
    caller must serialise them (the capture thread's frame observer, say).
    update() never raises for a bad frame; it returns an invalid estimate that
    says why.
    """

    def __init__(self, params: MarkerParams | None = None) -> None:
        self.params = params if params is not None else MarkerParams()
        self._cv2 = _import_cv2()
        self._reset()

    def _reset(self) -> None:
        self._reference: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None
        self._limits: _BlobLimits | None = None
        self._spacing = math.nan
        self._coverage = math.nan
        # Per grid step, each reference dot's neighbour there; and per grid
        # axis, its neighbours on either side.
        self._neighbours = np.zeros((0, 0), dtype=np.intp)
        self._axes = np.zeros((0, 2, 0), dtype=np.intp)
        # Where each reference dot is now; NaN while it is not followed.
        self._positions = np.zeros((0, 2))
        self._velocity = np.zeros(2)
        self._last_time: float | None = None
        self._lost_cause: str | None = None

    @property
    def has_reference(self) -> bool:
        return self._reference is not None

    def set_reference(self, image: np.ndarray) -> int:
        """Zero on a frame of the unloaded gel and return how many markers it shows.

        Raises ValueError when the frame shows no usable marker grid. The
        tracker is then left without a reference, so update() reports invalid
        until a reference is set.
        """

        self._reset()
        problem = _image_problem(image)
        if problem:
            raise ValueError(problem)
        p = self.params
        height, width = image.shape[:2]
        # A dot's size is unknown until dots are found: find them with bounds
        # from the image size, then again with bounds from their own size,
        # which is what tracking uses.
        points, areas, _ = self._blobs(image, self._image_limits(height, width))
        if len(points) >= p.min_markers:
            area = float(np.median(areas))
            limits = _BlobLimits(
                kernel=_odd(4.0 * math.sqrt(area / math.pi) + 1.0),
                min_area=max(3.0, area / p.area_ratio),
                max_area=area * p.area_ratio,
            )
            points, _, coverage = self._blobs(image, limits)
        if len(points) < p.min_markers:
            raise ValueError(
                f"Found {len(points)} markers, need at least {p.min_markers}; is the gel lit, in view and unloaded?"
            )
        neighbour = _neighbour_distances(points)
        spacing = float(np.median(neighbour))
        # The grid's own steps are what re-identification is checked against,
        # so a reference without them could never be trusted again.
        shifts = (
            _lattice_shifts(points, spacing) if np.std(neighbour) <= _MAX_SPACING_CV * spacing else np.zeros((0, 2))
        )
        axes = _grid_axes(shifts)
        if len(shifts) < 4 or len(axes) == 0:
            raise ValueError(
                f"The {len(points)} dark blobs found do not form a regular marker grid; is the gel in view and unloaded?"
            )
        self._reference = points
        self._shape = (height, width)
        self._limits = limits
        self._spacing = spacing
        self._coverage = coverage
        self._neighbours = _neighbour_table(points, shifts, spacing)
        self._axes = self._neighbours[axes]
        self._positions = points.copy()
        return len(points)

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Marker centroids as an (N, 2) float array of (x, y) pixels, for tests and overlays.

        Uses the reference's marker size once a reference is set, else bounds
        from the image size.
        """

        problem = _image_problem(image)
        if problem:
            raise ValueError(problem)
        limits = self._limits if self._limits is not None else self._image_limits(*image.shape[:2])
        return self._blobs(image, limits)[0]

    def update(self, image: np.ndarray, timestamp: float, frame_number: int) -> ShearEstimate:
        """Measure one frame against the reference.

        ``timestamp`` is the frame's host monotonic time in seconds
        (TactileFrame.captured_monotonic); it bounds how long the track may go
        without a usable frame.
        """

        if self._reference is None:
            return self._invalid(timestamp, frame_number, "no reference; call set_reference on the unloaded gel")
        problem = _image_problem(image)
        if not problem and image.shape[:2] != self._shape:
            problem = f"image is {image.shape[1]}x{image.shape[0]}, the reference is {self._shape[1]}x{self._shape[0]}"
        if not problem and not math.isfinite(timestamp):
            problem = f"timestamp {timestamp!r} is not finite"
        if problem:
            return self._invalid(timestamp, frame_number, problem)
        p = self.params
        if self._lost_cause is None and self._last_time is not None:
            elapsed = timestamp - self._last_time
            if not 0.0 <= elapsed <= p.max_gap_s:
                # Unwatched, the gel may have moved a whole spacing.
                self._lost_cause = f"{elapsed:.2f} s since the last tracked frame, limit {p.max_gap_s:.2f} s"
        detected, _, coverage = self._blobs(image, self._limits)
        if self._lost_cause is None:
            unusable = ""
            if len(detected) < p.min_markers:
                unusable = f"only {len(detected)} markers detected, need {p.min_markers}; is the gel lit and in view?"
            elif coverage > _MAX_COVERAGE_RATIO * self._coverage + _MAX_COVERAGE_EXCESS:
                unusable = (
                    f"dark blobs cover {coverage:.0%} of the frame against {self._coverage:.0%} in the reference; "
                    "it does not look like the gel"
                )
            if unusable:
                # A frame without usable dots (lens cap, glare, a corrupt
                # decode) says nothing about the gel, so the track waits for
                # the next one; the gap check bounds how long it may wait.
                return self._invalid(timestamp, frame_number, unusable)
            try:
                matched, columns = self._follow(detected)
            except _Lost as lost:
                self._lost_cause = str(lost)
            else:
                return self._commit(timestamp, frame_number, matched, detected[columns])
        try:
            matched, columns = self._reacquire(detected)
        except _Lost as advice:
            return self._invalid(timestamp, frame_number, f"lost track ({self._lost_cause}); {advice}")
        self._lost_cause = None
        self._velocity = np.zeros(2)
        return self._commit(timestamp, frame_number, matched, detected[columns])

    def _image_limits(self, height: int, width: int) -> _BlobLimits:
        p = self.params
        pixels = float(height * width)
        return _BlobLimits(
            # Comfortably wider than the largest dot max_area_fraction allows.
            kernel=_odd(min(height, width) / 12.0),
            min_area=max(3.0, p.min_area_fraction * pixels),
            max_area=p.max_area_fraction * pixels,
        )

    def _blobs(self, image: np.ndarray, limits: _BlobLimits) -> tuple[np.ndarray, np.ndarray, float]:
        """Dark round blobs, and how much of the image is dark.

        Returns (N, 2) contrast-weighted centroids (x, y), their (N,) areas in
        pixels, and the share of the image dark enough to belong to a blob.
        """

        cv2 = self._cv2
        p = self.params
        gray = cv2.GaussianBlur(cv2.transform(image.astype(np.float32), _CHANNEL_MEAN), (5, 5), 1.0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (limits.kernel, limits.kernel))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        # Relative darkness, so the LED gradient and broad shading divide out.
        contrast = (background - gray) / np.maximum(background, _MIN_BACKGROUND)
        mask = contrast > p.min_contrast
        coverage = float(np.count_nonzero(mask)) / mask.size
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.view(np.uint8), connectivity=8)
        if count <= 1:
            return np.zeros((0, 2)), np.zeros(0), coverage
        height, width = mask.shape
        index = np.flatnonzero(mask)
        label = labels.ravel()[index]
        x = (index % width).astype(np.float64)
        y = (index // width).astype(np.float64)
        # Weights fall to zero at the threshold, so pixels flickering across it
        # barely move the centroid.
        weight = contrast.ravel()[index].astype(np.float64) - p.min_contrast

        def total(values: np.ndarray) -> np.ndarray:
            return np.bincount(label, values, minlength=count)

        area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
        mean_x = total(x) / area
        mean_y = total(y) / area
        # Shape moments; the 1/12 is a pixel's own extent, so a one-pixel
        # streak still has a finite axis ratio.
        var_x = total(x * x) / area - mean_x**2 + 1.0 / 12.0
        var_y = total(y * y) / area - mean_y**2 + 1.0 / 12.0
        cov = total(x * y) / area - mean_x * mean_y
        half_sum = 0.5 * (var_x + var_y)
        root = np.sqrt((0.5 * (var_x - var_y)) ** 2 + cov**2)
        elongation = np.sqrt((half_sum + root) / np.maximum(half_sum - root, 1e-12))
        left, top = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
        box_width, box_height = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
        # A dot cut by the image edge has its centroid pulled inwards, which
        # would read as motion.
        whole = (left > 0) & (top > 0) & (left + box_width < width) & (top + box_height < height)
        mass = total(weight)
        keep = (
            whole
            & (area >= limits.min_area)
            & (area <= limits.max_area)
            & (elongation <= p.max_elongation)
            & (area >= p.min_fill * box_width * box_height)
            & (mass > 0.0)
        )
        keep[0] = False
        centroids = np.stack((total(weight * x)[keep], total(weight * y)[keep]), axis=1) / mass[keep, None]
        return centroids, area[keep], coverage

    def _follow(self, detected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Carry the followed dots into this frame, or raise _Lost when that is not safe.

        Returns reference indices and the matching indices into ``detected``.
        """

        p = self.params
        spacing = self._spacing
        followed = np.flatnonzero(np.isfinite(self._positions[:, 0]))
        previous = self._positions[followed]
        rows, columns = _match(previous + self._velocity, detected, p.track_gate * spacing)
        if len(rows) < p.min_markers:
            raise _Lost(f"only {len(rows)} markers followed from the previous frame")
        steps = detected[columns] - previous[rows]
        lengths = np.hypot(steps[:, 0], steps[:, 1])
        limit = p.max_step * spacing
        moved = float(np.median(lengths))
        if moved > limit:
            raise _Lost(f"markers moved {moved:.1f} px in one frame, limit {limit:.1f} px")
        # A jump of about a whole spacing in one frame lands each dot on its
        # neighbour's spot and passes the gates as a small motion. Frame to
        # frame the steps of a gel are smooth; paired one spacing off, they
        # carry the printed grid's jitter, and pairing each dot with its
        # neighbour's detection would smooth them again. Every gated pair
        # counts here, so the dots set aside below cannot hide a jump.
        gated = np.full_like(self._positions, np.nan)
        gated[followed[rows]] = detected[columns]
        self._check_identities(self._positions, gated, "steps")
        # One dot outrunning the field is more likely a mismatch than a
        # measurement; if it is real, the fit below brings it back.
        steady = lengths <= limit
        rows, columns, steps = rows[steady], columns[steady], steps[steady]
        if len(rows) < p.min_markers:
            raise _Lost(f"only {len(rows)} markers followed from the previous frame")
        matched = followed[rows]
        fit = _fit_similarity(self._reference[matched], detected[columns])
        if not fit.rms <= p.max_residual * spacing:
            raise _Lost(
                f"markers do not move as one field, fit residual {fit.rms:.1f} px RMS, "
                f"limit {p.max_residual * spacing:.1f} px"
            )
        # Dots not followed (occluded, out of view, or set aside above) rejoin
        # where the fit puts them; the tight gate keeps a neighbour out.
        others = np.setdiff1d(np.arange(len(self._reference)), matched)
        extra_rows, extra_columns = _match(fit.apply(self._reference[others]), detected, p.reacquire_gate * spacing)
        free = ~np.isin(extra_columns, columns)
        matched = np.concatenate((matched, others[extra_rows[free]]))
        columns = np.concatenate((columns, extra_columns[free]))
        # The same test against the reference catches identities that went
        # one spacing off in a frame the step test could not judge.
        current = np.full_like(self._reference, np.nan)
        current[matched] = detected[columns]
        self._check_identities(self._reference, current, "displacements")
        self._velocity = np.median(steps, axis=0)
        return matched, columns

    def _check_identities(self, source: np.ndarray, current: np.ndarray, what: str) -> None:
        """Raise _Lost when the field from source to current is clearly smoother with every identity one step over."""

        p = self.params
        floor = _ALIAS_NOISE_FLOOR * self._spacing
        for own, shifted in _pairing_roughness(source, current, self._neighbours, self._axes, p.min_markers):
            if own > max(p.alias_ratio * shifted, floor):
                raise _Lost(
                    f"marker {what} are rough ({own:.2f} px) but smooth ({shifted:.2f} px) with every marker taken "
                    "for its neighbour, so the gel jumped about a spacing"
                )

    def _reacquire(self, detected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Match straight against the reference, which only succeeds near the unloaded state."""

        p = self.params
        spacing = self._spacing
        gate = p.reacquire_gate * spacing
        rows, columns = _match(self._reference, detected, gate)
        if len(rows) < max(p.min_markers, p.min_matched_fraction * len(self._reference)):
            raise _Lost(_LOST_ADVICE)
        fit = _fit_similarity(self._reference[rows], detected[columns])
        if not fit.rms <= p.max_residual * spacing:
            raise _Lost(_LOST_ADVICE)
        # A regular grid also lines up with itself one spacing over, so a gel
        # held sheared by a whole spacing passes the gate too. Accept only when
        # the displacement field is smooth as paired and clearly rougher with
        # every identity moved one step; on a grid too regular to tell, that
        # never happens and only a re-zero recovers.
        current = np.full_like(self._reference, np.nan)
        current[rows] = detected[columns]
        floor = _ALIAS_NOISE_FLOOR * spacing
        pairs = _pairing_roughness(self._reference, current, self._neighbours, self._axes, p.min_markers)
        if not pairs or any(not shifted > max(p.alias_ratio * own, floor) for own, shifted in pairs):
            raise _Lost(_AMBIGUOUS_ADVICE)
        return rows, columns

    def _commit(self, timestamp: float, frame_number: int, matched: np.ndarray, current: np.ndarray) -> ShearEstimate:
        self._positions = np.full_like(self._reference, np.nan)
        self._positions[matched] = current
        self._last_time = float(timestamp)
        return self._estimate(timestamp, frame_number, matched, current)

    def _estimate(self, timestamp: float, frame_number: int, matched: np.ndarray, current: np.ndarray) -> ShearEstimate:
        p = self.params
        reference = self._reference[matched]
        fraction = len(matched) / len(self._reference)
        fit = _fit_similarity(reference, current)
        shear = fit.target - fit.origin
        moved = current - reference
        # atan2(b, a) turns clockwise on screen; report the usual counter-clockwise sense.
        twist = -math.atan2(fit.b, fit.a)
        spread = math.hypot(fit.a, fit.b) - 1.0
        displacement = float(np.max(np.hypot(moved[:, 0], moved[:, 1])))
        limit = p.max_residual * self._spacing
        if not all(math.isfinite(value) for value in (*shear, twist, spread, displacement, fit.rms)):
            reason = "the marker fit is not finite"
        elif fraction < p.min_matched_fraction:
            reason = f"only {fraction:.0%} of reference markers matched, need {p.min_matched_fraction:.0%}"
        elif fit.rms > limit:
            reason = f"markers do not move as one field, fit residual {fit.rms:.1f} px RMS, limit {limit:.1f} px"
        else:
            return ShearEstimate(
                timestamp=float(timestamp),
                frame_number=int(frame_number),
                valid=True,
                reason="",
                shear_px=(float(shear[0]), float(shear[1])),
                twist_rad=twist,
                spread=spread,
                max_displacement_px=displacement,
                matched_fraction=fraction,
                marker_spacing_px=self._spacing,
            )
        return self._invalid(timestamp, frame_number, reason, fraction)

    def _invalid(
        self, timestamp: float, frame_number: int, reason: str, matched_fraction: float = 0.0
    ) -> ShearEstimate:
        return ShearEstimate(
            timestamp=float(timestamp),
            frame_number=int(frame_number),
            valid=False,
            reason=reason,
            shear_px=(math.nan, math.nan),
            twist_rad=math.nan,
            spread=math.nan,
            max_displacement_px=math.nan,
            matched_fraction=float(matched_fraction),
            marker_spacing_px=self._spacing,
        )
