import json
from datetime import datetime
import numpy as np
import pickle
from typing import Any, Tuple, Optional
from .channel import ChannelDef

try: # for torch libraries
    import torch
except ImportError:
    torch = None

# returns (dtype_str, serialized_value)
def resolve_value(channel_def: Optional[ChannelDef], value: Any) -> Tuple[str, Any]:
    # If serializer exits
    if channel_def and channel_def.serializer is not None:
        value = channel_def.serializer(value)
        # pickle has to do both json and pickle serialization.
        if channel_def.pickle:
            json_payload = json.dumps(value, default=_json_default)
            pickle_payload = pickle.dumps(value)
            return "json", {"json": json_payload, "pickle": pickle_payload}

    # torch.Tensor
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    # numpy array
    if isinstance(value, np.ndarray):
        dtype_str = channel_def.dtype if channel_def else _map_numpy_dtype(value.dtype)
        return dtype_str, value.tobytes()

    # Python scalars
    if isinstance(value, bool):
        return "bool", value
    if isinstance(value, (int, np.integer)):
        return "i64", int(value)
    if isinstance(value, (float, np.floating)):
        return "f64", float(value)
    if isinstance(value, complex):
        # msgpack doesn't support complex, store as bytes (c128 = 2x f64)
        return "c128", np.array(value, dtype=np.complex128).tobytes()

    # strings
    if isinstance(value, str):
        return "str", value

    # dicts and lists -> JSON
    if isinstance(value, (dict, list)):
        return "json", json.dumps(value, default=_json_default)
    
    # raw bytes
    if isinstance(value, bytes):
        return "bytes", value

    # datetimes
    if isinstance(value, datetime):
        # ISO8601 string
        return "datetime", value.isoformat()

    # no match
    if channel_def:
        raise TypeError(
            f"Channel '{channel_def.name}': cannot auto-serialize {type(value).__name__}. "
            f"Register a serializer with declare_channel(..., serializer=fn)."
        )
    else:
        raise TypeError(f"Cannot auto-serialize {type(value).__name__} for ad-hoc logging.")

def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def _map_numpy_dtype(np_dtype) -> str:
    mapping = {
        np.float16: "f16",
        np.float32: "f32",
        np.float64: "f64",
        np.int8: "i8",
        np.int16: "i16",
        np.int32: "i32",
        np.int64: "i64",
        np.uint8: "u8",
        np.uint16: "u16",
        np.uint32: "u32",
        np.uint64: "u64",
        np.complex64: "c64",
        np.complex128: "c128",
        np.bool_: "bool"
    }
    return mapping.get(np_dtype.type, "bytes")
