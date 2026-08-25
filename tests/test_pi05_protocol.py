import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from fr3_pi05.protocol import OpenPiWebsocketClient, OpenPiZmqClient, packb, unpackb
from fr3_pi05.remote.serve_pi05_zmq import (
    handle_request,
    load_policy,
    resolve_action_expert_variant,
    warm_up,
)


class Pi05ProtocolTest(unittest.TestCase):
    def test_wine_variant_is_inferred_from_checkpoint_name(self) -> None:
        self.assertEqual(
            resolve_action_expert_variant("/models/pi05_wine_hybrid_17500", "auto"),
            "gemma_300m_lora",
        )
        self.assertEqual(
            resolve_action_expert_variant("/models/pi05_wine_aefull_17500", "auto"),
            "gemma_300m",
        )
        with self.assertRaisesRegex(RuntimeError, "Cannot determine"):
            resolve_action_expert_variant("/models/checkpoint_17500", "auto")

    def test_numpy_msgpack_round_trip(self) -> None:
        original = {
            "image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
            "state": np.array([1.0, 2.0], dtype=np.float32),
            "scalar": np.float64(3.5),
            "prompt": "test",
        }
        decoded = unpackb(packb(original))
        np.testing.assert_array_equal(decoded["image"], original["image"])
        np.testing.assert_array_equal(decoded["state"], original["state"])
        self.assertEqual(decoded["state"].dtype, np.float32)
        self.assertEqual(decoded["scalar"], 3.5)
        self.assertEqual(decoded["prompt"], "test")

    def test_object_arrays_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            packb(np.array([object()], dtype=object))

    def test_client_matches_openpi_server_message_order(self) -> None:
        from websockets.sync.server import serve

        received = {}

        def handler(connection) -> None:
            connection.send(packb({"checkpoint": "pi05_droid"}))
            received.update(unpackb(connection.recv()))
            connection.send(packb({"actions": np.zeros((15, 8), dtype=np.float32)}))

        with serve(handler, "127.0.0.1", 0, compression=None, max_size=None) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            port = server.socket.getsockname()[1]
            client = OpenPiWebsocketClient("127.0.0.1", port)
            response = client.infer({"prompt": "test", "state": np.zeros(7, dtype=np.float32)})
            client.close()
            server.shutdown()
            thread.join(timeout=2)

        self.assertEqual(client.metadata["checkpoint"], "pi05_droid")
        self.assertEqual(received["prompt"], "test")
        self.assertEqual(response["actions"].shape, (15, 8))

    def test_zmq_client_round_trip(self) -> None:
        import zmq

        socket = zmq.Context.instance().socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        port = socket.bind_to_random_port("tcp://127.0.0.1")

        def server() -> None:
            metadata_request = unpackb(socket.recv())
            self.assertEqual(metadata_request["operation"], "metadata")
            socket.send(packb({"success": True, "metadata": {"checkpoint": "pi05_droid"}}))
            inference_request = unpackb(socket.recv())
            self.assertEqual(inference_request["operation"], "infer")
            socket.send(
                packb({"success": True, "result": {"actions": np.zeros((15, 8), dtype=np.float32)}})
            )

        thread = threading.Thread(target=server)
        thread.start()
        client = OpenPiZmqClient("127.0.0.1", port, timeout_ms=2_000)
        response = client.infer({"prompt": "test"})
        client.close()
        thread.join(timeout=2)
        socket.close(linger=0)

        self.assertEqual(client.metadata["checkpoint"], "pi05_droid")
        self.assertEqual(response["actions"].shape, (15, 8))

    def test_zmq_client_can_bind_for_reverse_connection(self) -> None:
        import zmq

        probe = zmq.Context.instance().socket(zmq.REP)
        port = probe.bind_to_random_port("tcp://127.0.0.1")
        probe.close(linger=0)

        def server() -> None:
            socket = zmq.Context.instance().socket(zmq.REP)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(f"tcp://127.0.0.1:{port}")
            request = unpackb(socket.recv())
            self.assertEqual(request["operation"], "metadata")
            socket.send(packb({"success": True, "metadata": {"direction": "reverse"}}))
            socket.close(linger=0)

        thread = threading.Thread(target=server)
        thread.start()
        client = OpenPiZmqClient(
            "127.0.0.1",
            port,
            timeout_ms=2_000,
            connection_mode="bind",
        )
        client.close()
        thread.join(timeout=2)

        self.assertEqual(client.metadata["direction"], "reverse")

    def test_zmq_server_handler_invokes_policy(self) -> None:
        class FakePolicy:
            def infer(self, observation):
                self.observation = observation
                return {"actions": np.zeros((15, 8), dtype=np.float32)}

        policy = FakePolicy()
        metadata = handle_request(policy, {"name": "pi05_droid"}, {"operation": "metadata"})
        result = handle_request(
            policy,
            {"name": "pi05_droid"},
            {"operation": "infer", "observation": {"prompt": "move the block"}},
        )
        self.assertTrue(metadata["success"])
        self.assertEqual(metadata["metadata"]["name"], "pi05_droid")
        self.assertTrue(result["success"])
        self.assertEqual(policy.observation["prompt"], "move the block")
        self.assertIn("server_timing", result["result"])

    def test_zmq_server_rejects_untrained_wine_prompt(self) -> None:
        class FakePolicy:
            def infer(self, _observation):
                raise AssertionError("invalid prompt must not reach the policy")

        result = handle_request(
            FakePolicy(),
            {"tasks": ["pour gin into the jigger"]},
            {"operation": "infer", "observation": {"prompt": "pour the wine"}},
        )

        self.assertFalse(result["success"])
        self.assertIn("exactly match", result["error"])

    def test_custom_loader_uses_checkpoint_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            (checkpoint / "serve_custom_droid.py").write_text(
                """
DEFAULT_PROMPT = "custom task"
class Model:
    action_horizon = 16
MODEL = Model()
class Args:
    def __init__(self, checkpoint_dir, default_prompt):
        self.checkpoint_dir = checkpoint_dir
        self.default_prompt = default_prompt
def build_policy(args):
    return {"checkpoint": args.checkpoint_dir, "prompt": args.default_prompt}
""",
                encoding="utf-8",
            )

            policy, metadata = load_policy("custom_droid", "unused", str(checkpoint), None)

        self.assertEqual(policy["checkpoint"], str(checkpoint))
        self.assertEqual(policy["prompt"], "custom task")
        self.assertEqual(metadata["action_horizon"], 16)
        self.assertEqual(metadata["default_prompt"], "custom task")

    def test_server_warmup_uses_droid_contract(self) -> None:
        class FakePolicy:
            def infer(self, observation):
                self.observation = observation
                return {"actions": np.zeros((15, 8), dtype=np.float32)}

        policy = FakePolicy()
        elapsed = warm_up(policy, "test task")

        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(policy.observation["prompt"], "test task")
        self.assertEqual(policy.observation["observation/joint_position"].shape, (7,))
        self.assertEqual(policy.observation["observation/wrist_image_left"].shape, (224, 224, 3))

    def test_wine_loader_uses_training_side_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loader_path = root / "serve_wine.py"
            loader_path.write_text(
                """
STATE_HISTORY_LAGS = (45, 75)
NUM_STATE_FRAMES = 3
DEFAULT_ASSET_ID = "fitz0401/franka_pour_wine"
TASKS = ("pour lillet into the jigger", "pour gin into the jigger")
class Model:
    action_horizon = 16
def _model(action_expert_variant):
    return Model()
def build_policy(checkpoint, action_expert_variant, asset_id, use_exterior2=False):
    return {"checkpoint": checkpoint, "variant": action_expert_variant, "asset_id": asset_id, "use_exterior2": use_exterior2}
""",
                encoding="utf-8",
            )

            policy, metadata = load_policy(
                "wine",
                "unused",
                str(root / "checkpoint"),
                None,
                str(loader_path),
            )

        self.assertEqual(policy["variant"], "gemma_300m_lora")
        self.assertEqual(metadata["model"], "pi05_wine_hybrid")
        self.assertEqual(metadata["action_horizon"], 16)
        self.assertEqual(metadata["joint_observation_shape"], [3, 7])
        self.assertEqual(metadata["image_observation_shape"], [180, 320, 3])
        self.assertEqual(metadata["asset_id"], "fitz0401/franka_pour_wine")
        self.assertEqual(metadata["tasks"], [])
        self.assertFalse(metadata["uses_exterior2"])

    def test_wine_loader_accepts_new_dataset_asset_and_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loader_path = root / "serve_wine.py"
            loader_path.write_text(
                """
STATE_HISTORY_LAGS = (45, 75)
NUM_STATE_FRAMES = 3
DEFAULT_ASSET_ID = "old/dataset"
TASKS = ("old task",)
class Model:
    action_horizon = 16
def _model(action_expert_variant):
    return Model()
def build_policy(checkpoint, action_expert_variant, asset_id, use_exterior2=False):
    return {"asset_id": asset_id, "use_exterior2": use_exterior2}
""",
                encoding="utf-8",
            )

            policy, metadata = load_policy(
                "wine",
                "unused",
                str(root / "checkpoint"),
                None,
                str(loader_path),
                asset_id="fitz0401/new_dataset",
                tasks=["new task"],
            )

        self.assertEqual(policy["asset_id"], "fitz0401/new_dataset")
        self.assertEqual(metadata["asset_id"], "fitz0401/new_dataset")
        self.assertEqual(metadata["tasks"], ["new task"])

    def test_wine_loader_can_enable_second_exterior_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loader_path = root / "serve_wine.py"
            loader_path.write_text(
                """
STATE_HISTORY_LAGS = (45, 75)
NUM_STATE_FRAMES = 3
DEFAULT_ASSET_ID = "owner/dataset"
class Model:
    action_horizon = 16
def _model(action_expert_variant):
    return Model()
def build_policy(checkpoint, action_expert_variant, asset_id, use_exterior2=False):
    return {"use_exterior2": use_exterior2}
""",
                encoding="utf-8",
            )

            policy, metadata = load_policy(
                "wine",
                "unused",
                str(root / "checkpoint"),
                None,
                str(loader_path),
                use_exterior2=True,
            )

        self.assertTrue(policy["use_exterior2"])
        self.assertTrue(metadata["uses_exterior2"])

    def test_server_warmup_uses_wine_history_contract(self) -> None:
        class FakePolicy:
            def infer(self, observation):
                self.observation = observation
                return {"actions": np.zeros((15, 8), dtype=np.float32)}

        policy = FakePolicy()
        warm_up(policy, "pour the wine", (0, 45, 75))

        self.assertEqual(policy.observation["observation/joint_position"].shape, (3, 7))
        self.assertEqual(policy.observation["observation/gripper_position"].shape, (3,))
        self.assertEqual(policy.observation["observation/wrist_image_left"].shape, (180, 320, 3))

        with self.assertRaisesRegex(RuntimeError, "expected 16"):
            warm_up(policy, "pour the wine", (0, 45, 75), action_horizon=16)

    def test_server_warmup_includes_exterior2_when_enabled(self) -> None:
        class FakePolicy:
            def infer(self, observation):
                self.observation = observation
                return {"actions": np.zeros((16, 8), dtype=np.float32)}

        policy = FakePolicy()
        warm_up(policy, "pour the wine", (0, 45, 75), action_horizon=16, use_exterior2=True)

        self.assertEqual(
            policy.observation["observation/exterior_image_2_left"].shape,
            (180, 320, 3),
        )


if __name__ == "__main__":
    unittest.main()
