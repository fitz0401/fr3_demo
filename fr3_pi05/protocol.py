"""Minimal client for OpenPI's NumPy-over-msgpack WebSocket protocol."""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np


class OpenPiProtocolError(RuntimeError):
    """OpenPI server sent an error or a protocol-incompatible message."""


def _pack_numpy(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported NumPy dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _unpack_numpy(value: dict[bytes, Any]) -> Any:
    if b"__ndarray__" in value:
        return np.ndarray(buffer=value[b"data"], dtype=np.dtype(value[b"dtype"]), shape=value[b"shape"])
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def packb(value: Any) -> bytes:
    return msgpack.packb(value, default=_pack_numpy)


def unpackb(value: bytes) -> Any:
    return msgpack.unpackb(value, object_hook=_unpack_numpy)


class OpenPiWebsocketClient:
    """Protocol-compatible subset of OpenPI's official client policy."""

    def __init__(self, host: str, port: int) -> None:
        try:
            from websockets.sync.client import connect
        except ImportError as error:
            raise RuntimeError("pi0.5 network support is missing; run: pip install -e '.[pi05]'") from error
        uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"
        self._connection = connect(uri, compression=None, max_size=None, open_timeout=8, proxy=None)
        metadata = self._connection.recv()
        if isinstance(metadata, str):
            raise OpenPiProtocolError(f"OpenPI server rejected connection: {metadata}")
        decoded_metadata = unpackb(metadata)
        if not isinstance(decoded_metadata, dict):
            raise OpenPiProtocolError(
                f"OpenPI metadata was {type(decoded_metadata).__name__}, expected a dictionary"
            )
        self.metadata: dict[str, Any] = decoded_metadata

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self._connection.send(packb(observation))
        response = self._connection.recv()
        if isinstance(response, str):
            raise OpenPiProtocolError(f"OpenPI inference server error:\n{response}")
        result = unpackb(response)
        if not isinstance(result, dict):
            raise OpenPiProtocolError(f"OpenPI returned {type(result).__name__}, expected a dictionary")
        return result

    def close(self) -> None:
        self._connection.close()


class OpenPiZmqClient:
    """Request/reply client for the direct FR3 pi0.5 ZMQ server wrapper."""

    def __init__(self, host: str, port: int, timeout_ms: int = 60_000) -> None:
        try:
            import zmq
        except ImportError as error:
            raise RuntimeError("pi0.5 ZMQ support is missing; run: pip install -e '.[pi05]'") from error
        self._zmq = zmq
        self._socket = zmq.Context.instance().socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDTIMEO, 5_000)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.connect(f"tcp://{host}:{port}")
        response = self._request({"operation": "metadata"})
        metadata = response.get("metadata")
        if not isinstance(metadata, dict):
            raise OpenPiProtocolError("ZMQ server returned invalid policy metadata")
        self.metadata: dict[str, Any] = metadata

    def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            self._socket.send(packb(request))
            payload = self._socket.recv()
        except self._zmq.Again as error:
            raise TimeoutError("Timed out communicating with the pi0.5 ZMQ server") from error
        response = unpackb(payload)
        if not isinstance(response, dict):
            raise OpenPiProtocolError(f"ZMQ server returned {type(response).__name__}, expected a dictionary")
        if not response.get("success", False):
            raise OpenPiProtocolError(str(response.get("error", "Unknown pi0.5 ZMQ server error")))
        return response

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        response = self._request({"operation": "infer", "observation": observation})
        result = response.get("result")
        if not isinstance(result, dict):
            raise OpenPiProtocolError("ZMQ server returned an invalid inference result")
        return result

    def close(self) -> None:
        self._socket.close(linger=0)
