"""Hand guiding through Bamboo's guide mode, without the Pilot's guiding buttons."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

LOG = logging.getLogger(__name__)

GUIDE_FEATURE = "guide_mode"
# Holds the arm leaves on its own (idle) or that are part of normal operation
# (stopping); every other hold needs the operator to resume.
SILENT_HOLDS = {"none", "idle", "stopping"}
# How Bamboo answers once its guide loop has ended, on a Franka error or a stop.
GUIDE_ENDED = ("Guide mode stopped", "Guide mode is not running")
# libfranka's words for why a motion ended or was refused, and what the
# operator does about it. More specific markers come first.
FRANKA_ADVICE = (
    (
        "user stop pressed",
        "The user stop was pressed. Release it, check Desk is still in Execution mode, then run fr3-collect again.",
    ),
    (
        "user stopped",
        (
            "Franka is not in Execution mode: white lights mean Desk is in Programming mode, otherwise the user "
            "stop is pressed. Switch Desk to Execution and release the user stop, then run fr3-collect again."
        ),
    ),
    (
        "reflex",
        "Franka stopped the arm on a reflex, such as a collision. Run fr3-collect again; guide mode clears the reflex as it starts.",
    ),
)


def guide_failure_advice(error: str) -> str | None:
    """Say what to do about a raw libfranka error; Bamboo's own messages already do."""

    if "libfranka" not in error:
        return None
    lowered = error.lower()
    return next((advice for marker, advice in FRANKA_ADVICE if marker in lowered), None)


def describe_guide_failure(context: str, error: str) -> str:
    advice = guide_failure_advice(error)
    return f"{context}: {error}" + (f"\n{advice}" if advice else "")


def choose_guide_mode(robot: Any, force_pilot: bool) -> str:
    """Return "impedance" when Bamboo can float the arm itself, else "pilot".

    "pilot" is the original behaviour: nothing is commanded and the operator
    holds the robot's own guiding buttons. It is what an older Bamboo server,
    or ``--pilot-guide``, gets.
    """

    if force_pilot:
        return "pilot"
    return "impedance" if robot.supports(GUIDE_FEATURE) else "pilot"


class GuideHeartbeat:
    """Keep guide mode alive from its own thread and its own Bamboo connection.

    The collection loop blocks for about a second on every gripper command,
    longer than the server's watchdog, so the updates cannot come from it. If
    this thread stops for any reason, the updates stop and the arm damps to a
    stop and holds, which is the safe direction.

    Without a gel monitor every update says level 0: keep floating, no wall.
    With one, each update carries the gel's latest reading: its level, the
    direction that loads it, and its frame sequence number. A repeated
    sequence number is not fresh for Bamboo, so a stalled GelSight trips the
    watchdog, and an invalid reading holds the arm.
    """

    def __init__(self, client_factory: Callable[[], Any], rate_hz: float = 30.0, gel: Any | None = None) -> None:
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self._gel = gel
        self._client_factory = client_factory
        self._period = 1.0 / rate_hz
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._status: dict[str, Any] = {}

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Guide heartbeat already started")
        self._thread = threading.Thread(target=self._loop, name="guide-heartbeat", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        client = None
        try:
            client = self._client_factory()
            while not self._stop.is_set():
                started = time.monotonic()
                if self._gel is None:
                    response = client.update_guiding(level=0.0, valid=True)
                else:
                    reading = self._gel.latest()
                    response = client.update_guiding(
                        level=reading.level if reading.valid else 0.0,
                        valid=reading.valid,
                        direction=list(reading.direction_ee) if reading.direction_ee is not None else None,
                        seq=reading.seq,
                    )
                if not response.get("success", False):
                    raise RuntimeError(str(response.get("error", "Bamboo rejected a guide update")))
                with self._lock:
                    self._status = dict(response)
                self._stop.wait(max(0.0, self._period - (time.monotonic() - started)))
        except BaseException as error:  # noqa: BLE001 - surfaced through check_health
            with self._lock:
                self._error = error
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception as error:  # noqa: BLE001 - closing must not mask the original error
                    LOG.warning("Closing the guide heartbeat connection failed: %s", error)

    @property
    def ended_on_robot(self) -> bool:
        """Whether the heartbeat stopped because Bamboo's guide loop had ended."""

        with self._lock:
            error = self._error
        return error is not None and str(error).startswith(GUIDE_ENDED)

    def check_health(self) -> None:
        with self._lock:
            error = self._error
        if error is None:
            return
        if self.ended_on_robot:
            # Franka stopped the motion (or it was stopped): the arm no longer floats.
            raise RuntimeError(describe_guide_failure("Guide mode ended, so the arm no longer floats", str(error)))
        raise RuntimeError(f"Guide heartbeat stopped, so the arm is holding: {error}") from error

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def hold_message(state: dict[str, Any], gel_reason: str | None = None) -> str | None:
    """Describe a hold the operator has to act on, or return None."""

    if state.get("guide_phase") in (None, "free", "release"):
        return None
    reason = state.get("guide_hold_reason", "none")
    if reason in SILENT_HOLDS:
        return None
    advice = {
        "watchdog": "guide updates stopped arriving",
        "gel_invalid": "the gel signal is not trustworthy"
        + (f" ({gel_reason}); B opens the gripper, which unloads the gel and re-zeroes it" if gel_reason else ""),
        "gel_limit": "the gel reached its load limit; back off",
        "joint_limit": "a joint is at its limit; move it back in range",
    }.get(reason, reason)
    return f"Arm holding: {advice}. Press RB to resume."
