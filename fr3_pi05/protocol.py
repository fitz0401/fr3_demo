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
