import threading
import unittest

import numpy as np

from fr3_pi05.protocol import OpenPiWebsocketClient, packb, unpackb


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


if __name__ == "__main__":
    unittest.main()
