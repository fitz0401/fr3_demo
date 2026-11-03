"""Gel shear from the marker field, on synthetic GelSight frames.

Frames are drawn the way the sensor sees the gel: dark dots on a jittered grid
over a background lit in different colours from different sides, drawn at 4x
and reduced with INTER_AREA as the capture does, then blurred and given sensor
noise. The grid reaches past every edge, so dots can slide into view.
"""

import math
import time
import unittest
from functools import lru_cache

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - the [tactile] extra installs it
    cv2 = None

from fr3_demo.tactile_shear import MarkerParams, MarkerShearTracker, ShearEstimate

requires_cv2 = unittest.skipIf(cv2 is None, "needs OpenCV: pip install -e '.[tactile]'")

HEIGHT, WIDTH = 240, 320
SPACING = 26.0
RADIUS = 3.6
CENTRE = np.array([(WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0])
SUPERSAMPLE = 4
FRAME_PERIOD = 1.0 / 25.0


def _grid(jitter: float = 1.2, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xs = np.arange(-3 * SPACING, WIDTH + 3 * SPACING, SPACING) + 7.3
    ys = np.arange(-3 * SPACING, HEIGHT + 3 * SPACING, SPACING) + 5.6
    points = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return points + rng.uniform(-jitter, jitter, points.shape)


def _moved(
    points: np.ndarray, shift: tuple[float, float] = (0.0, 0.0), twist: float = 0.0, scale: float = 1.0
) -> np.ndarray:
    """The dots after a similarity about the image centre; twist turns counter-clockwise as displayed."""

    c, s = math.cos(twist), math.sin(twist)
    offset = points - CENTRE
    # y points down, so counter-clockwise on screen takes +x towards -y.
    turned = np.stack((c * offset[:, 0] + s * offset[:, 1], -s * offset[:, 0] + c * offset[:, 1]), axis=1)
    return CENTRE + scale * turned + np.asarray(shift, dtype=np.float64)


@lru_cache(maxsize=1)
def _background() -> np.ndarray:
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    u, v = x / WIDTH, y / HEIGHT
    return np.stack((110 + 100 * u, 205 - 80 * v, 120 + 90 * (1 - u) * v), axis=-1)


def _bump(x0: float, y0: float, sigma: float) -> np.ndarray:
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


def _render(points, *, hidden=None, shading=None, noise: float = 2.0, seed: int = 0) -> np.ndarray:
    scale = SUPERSAMPLE
    cover = np.zeros((HEIGHT * scale, WIDTH * scale), np.uint8)
    radius = round(RADIUS * scale * 16)
    for x, y in points:
        if not (-10 < x < WIDTH + 10 and -10 < y < HEIGHT + 10) or (hidden is not None and hidden(x, y)):
            continue
        # Four fractional bits; a reduced pixel's centre x sits at scale * x + (scale - 1) / 2.
        centre = (round((x * scale + (scale - 1) / 2) * 16), round((y * scale + (scale - 1) / 2) * 16))
        cv2.circle(cover, centre, radius, 255, thickness=-1, lineType=cv2.LINE_AA, shift=4)
    cover = cv2.resize(cover, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    image = _background() * (1.0 - 0.65 * cover)[..., None]
    if shading is not None:
        image = image * shading[..., None]
    image = cv2.GaussianBlur(image, (0, 0), 0.7)
    image = image + np.random.default_rng(seed).normal(0.0, noise, image.shape)
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def _lens(points: np.ndarray, k: float) -> np.ndarray:
    """Barrel distortion, as a wide lens draws the gel's grid in towards the corners."""

    offset = (points - CENTRE) / 200.0
    return CENTRE + 200.0 * offset * (1.0 - k * np.sum(offset**2, axis=1, keepdims=True))


def _every_other(grid: np.ndarray) -> np.ndarray:
    """Keeps every other row and column of dots, so no kept dot has a kept grid neighbour."""

    columns = len(np.arange(-3 * SPACING, WIDTH + 3 * SPACING, SPACING))
    index = np.arange(len(grid))
    return (index // columns % 2 == 0) & (index % columns % 2 == 0)


def _nearest(found: np.ndarray, truth: np.ndarray) -> np.ndarray:
    offset = found[:, None, :] - truth[None, :, :]
    return np.hypot(offset[..., 0], offset[..., 1]).min(axis=1)


def _clear_of_edges(points: np.ndarray, margin: float) -> int:
    x, y = points[:, 0], points[:, 1]
    return int(np.count_nonzero((x >= margin) & (x <= WIDTH - 1 - margin) & (y >= margin) & (y <= HEIGHT - 1 - margin)))


class _Gel:
    """A tracker zeroed on the unloaded grid, fed frames at the sensor's 25 Hz."""

    def __init__(self, grid: np.ndarray | None = None, params: MarkerParams | None = None) -> None:
        self.grid = _grid() if grid is None else grid
        self.tracker = MarkerShearTracker(params)
        self.tracker.set_reference(_render(self.grid, seed=1000))
        self.frame = 0

    def show(self, image: np.ndarray, frames_later: int = 1) -> ShearEstimate:
        self.frame += frames_later
        return self.tracker.update(image, self.frame * FRAME_PERIOD, self.frame)

    def pose(self, *, hidden=None, shading=None, frames_later: int = 1, **motion) -> ShearEstimate:
        image = _render(_moved(self.grid, **motion), hidden=hidden, shading=shading, seed=self.frame + 1)
        return self.show(image, frames_later)


def _shear_error(estimate: ShearEstimate, shift: tuple[float, float]) -> float:
    return math.hypot(estimate.shear_px[0] - shift[0], estimate.shear_px[1] - shift[1])


@requires_cv2
class DetectionTest(unittest.TestCase):
    def test_finds_each_whole_marker_to_a_fraction_of_a_pixel(self) -> None:
        grid = _grid()
        found = MarkerShearTracker().detect(_render(grid, seed=1))

        errors = _nearest(found, grid)
        self.assertLess(float(np.max(errors)), 0.1)
        self.assertLess(float(np.median(errors)), 0.05)
        # Dots cut by the image edge are left out: their centroid is pulled inwards.
        self.assertGreaterEqual(len(found), _clear_of_edges(grid, RADIUS + 3))
        self.assertLessEqual(len(found), _clear_of_edges(grid, RADIUS - 1))

    def test_the_reference_reports_its_marker_count(self) -> None:
        image = _render(_grid(), seed=1)
        tracker = MarkerShearTracker()

        count = tracker.set_reference(image)

        self.assertTrue(tracker.has_reference)
        self.assertEqual(count, len(tracker.detect(image)))
        self.assertGreater(count, 90)
        estimate = tracker.update(_render(_grid(), seed=2), 0.0, 0)
        # The median nearest-neighbour distance; jitter pulls it slightly under the pitch.
        self.assertAlmostEqual(estimate.marker_spacing_px, SPACING, delta=0.1 * SPACING)

    def test_contact_shading_does_not_move_the_markers(self) -> None:
        grid = _grid()
        # A bright and a dark lobe, as a pressed object shades the gel under coloured side light.
        shading = 1.0 + 0.5 * _bump(140, 110, 25) - 0.5 * _bump(175, 125, 25)
        plain = MarkerShearTracker().detect(_render(grid, seed=1))
        shaded = MarkerShearTracker().detect(_render(grid, shading=shading, seed=1))

        self.assertGreaterEqual(len(shaded), len(plain) - 2)
        self.assertLess(float(np.max(_nearest(shaded, grid))), 0.3)

    def test_a_frame_without_markers_has_none(self) -> None:
        frames = {
            "uniform": np.full((HEIGHT, WIDTH, 3), 128, np.uint8),
            "saturated": np.full((HEIGHT, WIDTH, 3), 255, np.uint8),
            "lens cap": np.random.default_rng(0).integers(0, 8, (HEIGHT, WIDTH, 3)).astype(np.uint8),
            "gel without dots": _render(np.zeros((0, 2)), seed=1),
        }
        for name, image in frames.items():
            with self.subTest(name):
                self.assertEqual(len(MarkerShearTracker().detect(image)), 0)


@requires_cv2
class ReferenceTest(unittest.TestCase):
    def test_no_reference_is_invalid(self) -> None:
        tracker = MarkerShearTracker()

        estimate = tracker.update(_render(_grid(), seed=1), 1.0, 7)

        self.assertFalse(tracker.has_reference)
        self.assertFalse(estimate.valid)
        self.assertIn("no reference", estimate.reason)
        self.assertEqual((estimate.timestamp, estimate.frame_number), (1.0, 7))

    def test_a_frame_without_a_marker_grid_cannot_be_the_reference(self) -> None:
        frames = {
            "uniform": np.full((HEIGHT, WIDTH, 3), 128, np.uint8),
            # Plenty of dark blobs, but not on a grid.
            "noise": np.random.default_rng(0).integers(0, 256, (HEIGHT, WIDTH, 3)).astype(np.uint8),
        }
        for name, image in frames.items():
            with self.subTest(name):
                tracker = MarkerShearTracker()
                with self.assertRaises(ValueError):
                    tracker.set_reference(image)
                self.assertFalse(tracker.has_reference)

    def test_a_failed_re_zero_leaves_no_reference(self) -> None:
        # Measuring against a reference the caller just tried to replace would
        # be a guess, so a failed re-zero fails closed.
        gel = _Gel()
        with self.assertRaises(ValueError):
            gel.tracker.set_reference(np.full((HEIGHT, WIDTH, 3), 128, np.uint8))

        self.assertFalse(gel.tracker.has_reference)
        self.assertFalse(gel.pose().valid)


@requires_cv2
class ShearTest(unittest.TestCase):
    def test_an_unloaded_frame_reads_zero(self) -> None:
        estimate = _Gel().pose()

        self.assertTrue(estimate.valid, estimate.reason)
        self.assertEqual(estimate.reason, "")
        self.assertLess(_shear_error(estimate, (0.0, 0.0)), 0.05)
        self.assertLess(abs(estimate.twist_rad), 1e-4)
        self.assertLess(abs(estimate.spread), 1e-4)
        self.assertLess(estimate.max_displacement_px, 0.2)
        self.assertGreater(estimate.matched_fraction, 0.95)

    def test_a_translation_is_measured_to_a_fraction_of_a_pixel(self) -> None:
        shift = (2.37, -1.61)
        estimate = _Gel().pose(shift=shift)

        self.assertTrue(estimate.valid, estimate.reason)
        self.assertLess(_shear_error(estimate, shift), 0.1)
        self.assertAlmostEqual(estimate.max_displacement_px, math.hypot(*shift), delta=0.2)

    def test_shear_past_a_whole_spacing_keeps_each_marker_identity(self) -> None:
        # Matched straight against the reference, every dot past half a
        # spacing would be taken for its neighbour; followed frame to frame,
        # it keeps its identity.
        gel = _Gel()
        total = (34.0, 9.0)
        for step in range(1, 13):
            shift = (total[0] * step / 12, total[1] * step / 12)
            estimate = gel.pose(shift=shift)
            self.assertTrue(estimate.valid, f"step {step}: {estimate.reason}")
            self.assertLess(_shear_error(estimate, shift), 0.1, f"step {step}")

        self.assertGreater(math.hypot(*estimate.shear_px), estimate.marker_spacing_px)

    def test_markers_slide_out_of_view_and_back(self) -> None:
        gel = _Gel()
        depth = 1.2 * SPACING
        out = [gel.pose(shift=(0.0, depth * step / 12)) for step in range(1, 13)]
        back = [gel.pose(shift=(0.0, depth * step / 12)) for step in range(11, -1, -1)]

        for estimate in out + back:
            self.assertTrue(estimate.valid, estimate.reason)
        self.assertLess(_shear_error(out[-1], (0.0, depth)), 0.1)
        # Only reference dots count: the row leaving at the bottom lowers the
        # fraction, and the row entering at the top does not make up for it.
        self.assertLess(out[-1].matched_fraction, 0.95)
        # Back in view, the reference dots rejoin where the fit puts them.
        self.assertGreater(back[-1].matched_fraction, 0.95)
        self.assertLess(_shear_error(back[-1], (0.0, 0.0)), 0.05)

    def test_twist_is_counter_clockwise_positive_as_displayed(self) -> None:
        for degrees in (3.0, -3.0):
            with self.subTest(degrees=degrees):
                gel = _Gel()
                for step in range(1, 7):
                    estimate = gel.pose(twist=math.radians(degrees * step / 6))
                self.assertTrue(estimate.valid, estimate.reason)
                self.assertAlmostEqual(math.degrees(estimate.twist_rad), degrees, delta=0.02)
                self.assertLess(abs(estimate.spread), 1e-3)

    def test_pressing_spreads_the_field(self) -> None:
        gel = _Gel()
        for step in range(1, 5):
            estimate = gel.pose(scale=1.0 + 0.01 * step)

        self.assertTrue(estimate.valid, estimate.reason)
        self.assertAlmostEqual(estimate.spread, 0.04, delta=1e-3)
        self.assertLess(abs(estimate.twist_rad), 1e-3)

    def test_a_third_of_the_markers_hidden_is_still_valid(self) -> None:
        gel = _Gel()
        gel.pose(shift=(1.0, 0.5))
        hidden = gel.pose(shift=(2.0, 1.0), hidden=lambda x, y: x < 0.3 * WIDTH)
        shown = gel.pose(shift=(3.0, 1.5))

        self.assertTrue(hidden.valid, hidden.reason)
        self.assertLess(hidden.matched_fraction, 0.8)
        self.assertLess(_shear_error(hidden, (2.0, 1.0)), 0.1)
        self.assertTrue(shown.valid, shown.reason)
        self.assertGreater(shown.matched_fraction, 0.95)
        self.assertLess(_shear_error(shown, (3.0, 1.5)), 0.1)

    def test_contact_shading_does_not_read_as_shear(self) -> None:
        shading = 1.0 + 0.5 * _bump(140, 110, 25) - 0.5 * _bump(175, 125, 25)
        estimate = _Gel().pose(shading=shading)

        self.assertTrue(estimate.valid, estimate.reason)
        self.assertLess(_shear_error(estimate, (0.0, 0.0)), 0.1)


@requires_cv2
class FailClosedTest(unittest.TestCase):
    def test_an_invalid_estimate_carries_no_numbers(self) -> None:
        gel = _Gel()
        estimate = gel.pose(shift=(0.6 * SPACING, 0.0))

        self.assertFalse(estimate.valid)
        for value in (*estimate.shear_px, estimate.twist_rad, estimate.spread, estimate.max_displacement_px):
            self.assertTrue(math.isnan(value))
        self.assertEqual(estimate.matched_fraction, 0.0)

    def test_a_frame_without_markers_is_invalid_and_the_track_waits(self) -> None:
        frames = {
            "uniform": np.full((HEIGHT, WIDTH, 3), 128, np.uint8),
            "saturated": np.full((HEIGHT, WIDTH, 3), 255, np.uint8),
            "lens cap": np.zeros((HEIGHT, WIDTH, 3), np.uint8),
        }
        for name, image in frames.items():
            with self.subTest(name):
                gel = _Gel()
                gel.pose(shift=(1.0, 0.0))
                blank = gel.show(image)
                after = gel.pose(shift=(2.0, 0.0))

                self.assertFalse(blank.valid)
                self.assertIn("markers detected", blank.reason)
                self.assertTrue(after.valid, after.reason)
                self.assertLess(_shear_error(after, (2.0, 0.0)), 0.1)

    def test_a_corrupt_frame_is_invalid_and_the_track_waits(self) -> None:
        gel = _Gel()
        gel.pose(shift=(1.0, 0.0))
        noise = gel.show(np.random.default_rng(0).integers(0, 256, (HEIGHT, WIDTH, 3)).astype(np.uint8))
        after = gel.pose(shift=(2.0, 0.0))

        self.assertFalse(noise.valid)
        self.assertIn("does not look like the gel", noise.reason)
        self.assertTrue(after.valid, after.reason)

    def test_a_frame_of_another_size_or_kind_is_invalid(self) -> None:
        gel = _Gel()
        small = gel.show(np.zeros((HEIGHT // 2, WIDTH // 2, 3), np.uint8))
        gray = gel.show(np.zeros((HEIGHT, WIDTH), np.uint8))
        after = gel.pose(shift=(1.0, 0.0))

        self.assertFalse(small.valid)
        self.assertIn(f"the reference is {WIDTH}x{HEIGHT}", small.reason)
        self.assertFalse(gray.valid)
        self.assertIn("expected", gray.reason)
        self.assertTrue(after.valid, after.reason)

    def test_a_jump_of_over_half_a_spacing_loses_the_track_until_unloaded(self) -> None:
        gel = _Gel()
        self.assertTrue(gel.pose().valid)
        jumped = [gel.pose(shift=(0.6 * SPACING, 0.0)) for _ in range(4)]
        unloaded = gel.pose()
        after = gel.pose(shift=(1.0, 0.0))

        for estimate in jumped:
            self.assertFalse(estimate.valid)
            self.assertIn("lost track", estimate.reason)
            self.assertIn("unload the gel or re-zero", estimate.reason)
        self.assertTrue(unloaded.valid, unloaded.reason)
        self.assertLess(_shear_error(unloaded, (0.0, 0.0)), 0.05)
        self.assertTrue(after.valid, after.reason)

    def test_a_jump_of_about_a_whole_spacing_is_not_read_as_a_small_motion(self) -> None:
        # Each dot lands near its neighbour's old spot, so the gates see a
        # small motion; the grid's irregularity gives the jump away.
        for spacings in (0.8, 1.0, 1.2):
            with self.subTest(spacings=spacings):
                gel = _Gel()
                gel.pose()
                estimate = gel.pose(shift=(spacings * SPACING, 0.0))
                self.assertFalse(estimate.valid)
                self.assertIn("lost track", estimate.reason)

    def test_a_gel_held_one_spacing_over_is_not_taken_for_unloaded(self) -> None:
        # Matched straight against the reference, the grid one spacing over
        # lines up with it; re-acquiring then would report a gel at its
        # tearing point as unloaded.
        for shift in ((SPACING, 0.0), (0.0, -SPACING), (SPACING, SPACING)):
            with self.subTest(shift=shift):
                gel = _Gel()
                gel.pose()
                gel.pose(shift=(0.6 * SPACING, 0.0))
                held = [gel.pose(shift=shift) for _ in range(3)]
                unloaded = gel.pose()

                for estimate in held:
                    self.assertFalse(estimate.valid)
                    self.assertIn("re-zero", estimate.reason)
                self.assertTrue(unloaded.valid, unloaded.reason)

    def test_a_jump_hidden_from_the_step_check_is_caught_against_the_reference(self) -> None:
        # With every other row and column hidden, no followed dot has a grid
        # neighbour, so the jump frame cannot be judged; back in full view,
        # the field against the reference gives the jump away.
        gel = _Gel()
        gel.pose()
        sparse = gel.show(_render(_moved(gel.grid[_every_other(gel.grid)], shift=(SPACING, 0.0)), seed=50))
        full = gel.pose(shift=(SPACING, 0.0))

        self.assertFalse(sparse.valid)
        self.assertNotIn("lost track", sparse.reason)
        self.assertFalse(full.valid)
        self.assertIn("lost track", full.reason)

    def test_lens_distortion_cannot_tell_a_regular_grid_from_itself_one_spacing_over(self) -> None:
        # A gel moved by a whole spacing images exactly like the unloaded gel,
        # distortion and all, so a lost track must not come back by itself.
        printed = _grid(jitter=0.0)
        for k in (0.05, 0.1):
            with self.subTest(k=k):
                tracker = MarkerShearTracker()
                tracker.set_reference(_render(_lens(printed, k), seed=1000))
                shifts = ((0.0, 0.0), (0.6 * SPACING, 0.0), (SPACING, 0.0), (0.0, 0.0))
                estimates = [
                    tracker.update(
                        _render(_lens(printed + np.asarray(shift), k), seed=frame), frame * FRAME_PERIOD, frame
                    )
                    for frame, shift in enumerate(shifts, 1)
                ]

                self.assertTrue(estimates[0].valid, estimates[0].reason)
                for estimate in estimates[1:]:
                    self.assertFalse(estimate.valid)
                self.assertIn("re-zero", estimates[2].reason)

    def test_a_gel_snapping_back_to_rest_is_reacquired_at_once(self) -> None:
        gel = _Gel()
        for step in range(1, 7):
            gel.pose(shift=(0.1 * SPACING * step, 0.0))
        snapped = gel.pose()

        self.assertTrue(snapped.valid, snapped.reason)
        self.assertLess(_shear_error(snapped, (0.0, 0.0)), 0.05)

    def test_a_perfectly_regular_grid_needs_a_re_zero_after_a_loss(self) -> None:
        # Nothing tells a dot from its neighbour, so even the unloaded gel
        # cannot be re-identified safely; only a new reference recovers.
        gel = _Gel(_grid(jitter=0.0))
        self.assertTrue(gel.pose().valid)
        gel.pose(shift=(0.6 * SPACING, 0.0))
        unloaded = gel.pose()

        self.assertFalse(unloaded.valid)
        self.assertIn("re-zero", unloaded.reason)
        gel.tracker.set_reference(_render(gel.grid, seed=2000))
        self.assertTrue(gel.pose().valid)

    def test_a_long_gap_between_frames_loses_the_track(self) -> None:
        gel = _Gel()
        self.assertTrue(gel.pose().valid)
        late = gel.pose(shift=(0.3 * SPACING, 0.0), frames_later=25)
        unloaded = gel.pose()

        self.assertFalse(late.valid)
        self.assertIn("since the last tracked frame", late.reason)
        self.assertTrue(unloaded.valid, unloaded.reason)

    def test_params_keep_every_gate_under_half_a_spacing(self) -> None:
        for name in ("track_gate", "max_step", "reacquire_gate"):
            with self.subTest(name), self.assertRaises(ValueError):
                MarkerParams(**{name: 0.5})
        with self.assertRaises(ValueError):
            MarkerParams(min_matched_fraction=0.0)
        MarkerParams()


@requires_cv2
class TimingTest(unittest.TestCase):
    def test_update_keeps_up_with_the_sensor(self) -> None:
        gel = _Gel()
        frames = [_render(_moved(gel.grid, shift=(0.3 * step, 0.1 * step)), seed=step) for step in range(1, 41)]
        took = []
        for frame in frames:
            started = time.perf_counter()
            estimate = gel.show(frame)
            took.append(time.perf_counter() - started)
            self.assertTrue(estimate.valid, estimate.reason)

        # About 1-3 ms on a desktop CPU; the bound only catches a gross regression.
        self.assertLess(float(np.median(took)), 0.02)


if __name__ == "__main__":
    unittest.main()
