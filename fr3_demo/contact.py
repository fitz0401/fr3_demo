"""Contact limits: what Bamboo's streams may do to the tool and the gel.

Bamboo enforces them in its 1 kHz loop (see Bamboo's README); this module
only describes them for the operator.
"""

from __future__ import annotations

from typing import Any


def describe_contact_limits(limits: dict[str, Any]) -> str:
    """One line for the operator, from the limits Bamboo echoes back."""

    def newtons(key: str) -> str:
        value = float(limits.get(key, 0.0))
        return f"{value:.0f} N" if value > 0 else "off"

    point = [round(float(value), 3) for value in limits.get("contact_point", [0.0, 0.0, 0.0])]
    parts = [f"push cap {newtons('max_force_n')}" + (" (shadow: reported only)" if limits.get("shadow") else "")]
    guard = newtons("guard_force_n")
    if float(limits.get("guard_hard_force_n", 0.0)) > 0 and guard != "off":
        guard += f", hard {newtons('guard_hard_force_n')}"
    parts.append(f"force guard {guard}")
    speed = float(limits.get("max_point_speed", 0.0))
    parts.append(f"tool speed {speed:.2f} m/s" if speed > 0 else "tool speed off")
    reflex = float(limits.get("collision_force_n", 0.0))
    parts.append(f"Franka reflex {reflex:.0f} N" if reflex > 0 else "Franka reflex default")
    return f"Contact limits at {point} m (flange frame): " + "; ".join(parts) + "."


def _number(state: dict[str, Any], key: str) -> float | None:
    value = state.get(key)
    return None if value is None else float(value)


def contact_status(state: dict[str, Any], peaks: dict[str, float]) -> str:
    """One status line from a Bamboo robot state, updating the running peaks."""

    wrench = state.get("ext_wrench")
    force = None if wrench is None else float(sum(value * value for value in wrench[:3]) ** 0.5)
    push = _number(state, "contact_push")
    scale = _number(state, "contact_scale")
    if force is not None:
        peaks["force"] = max(peaks.get("force", 0.0), force)
    if push is not None and push != float("inf"):
        peaks["push"] = max(peaks.get("push", 0.0), push)
    if scale is not None:
        peaks["scale"] = min(peaks.get("scale", 1.0), scale)

    def show(value: float | None, unit: str, peak_key: str, peak_word: str = "max") -> str:
        if value is None:
            return "n/a"
        return f"{value:5.1f}{unit} ({peak_word} {peaks.get(peak_key, value):.1f})"

    parts = [
        f"tool force {show(force, ' N', 'force')}",
        f"push {show(push, ' N', 'push')}",
        "cap n/a" if scale is None else f"cap {scale:.2f} (min {peaks.get('scale', scale):.2f})",
        f"guard {state.get('guard_trip', 'n/a')}",
        f"loop {state.get('loop_mode', 'n/a')}",
    ]
    if state.get("loop_mode") == "guide":
        parts.append(f"leash {float(state.get('guide_lead', 0.0)):.3f} rad")
        parts.append("ratchet on" if state.get("guide_ratchet") else "ratchet off")
    return " | ".join(parts)


def main(argv: list[str] | None = None) -> int:
    """Watch Franka's force estimate and Bamboo's contact limits live; sends no commands."""

    import argparse
    import time
    from pathlib import Path

    from fr3_demo.settings import default_config_path, load_config

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--server-ip")
    parser.add_argument("--control-port", type=int)
    parser.add_argument("--rate", type=float, default=10.0, help="status updates per second")
    args = parser.parse_args(argv)
    robot = load_config(args.config).get("robot", {})
    from bamboo import BambooFrankaClient

    client = BambooFrankaClient(
        server_ip=args.server_ip or robot.get("server_ip", "172.16.0.20"),
        control_port=args.control_port or int(robot.get("control_port", 5555)),
        enable_gripper=False,
    )
    capabilities = client.get_capabilities()
    limits = capabilities.get("contact_limits")
    if limits and limits.get("enabled"):
        print(describe_contact_limits(limits))
    else:
        print("No contact limits are set on Bamboo right now (they are set when a session with [contact] starts).")
    print("Tool force is Franka's external force estimate: the tool's load only while nobody touches the arm.")
    print("Ctrl-C to stop.\n")
    peaks: dict[str, float] = {}
    try:
        while True:
            line = contact_status(client.get_joint_states(), peaks)
            print("\r" + line + "   ", end="", flush=True)
            time.sleep(1.0 / max(args.rate, 1.0))
    except KeyboardInterrupt:
        print()
    finally:
        client.close()
    return 0
