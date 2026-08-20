import threading
import unittest

import numpy as np

from fr3_pi05.protocol import OpenPiWebsocketClient, OpenPiZmqClient, packb, unpackb
from fr3_pi05.remote.serve_pi05_zmq import handle_request


class Pi05ProtocolTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
