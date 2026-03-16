import json
from datetime import datetime
import numpy as np
import pickle
import struct
from typing import Any, Tuple, Optional, Callable
from .channel import ChannelDef

try: # for torch libraries
    import torch
except ImportError:
    torch = None

# specialized primitive handlers
def _handle_bool(val: bool) -> Tuple[str, int]:
    return "bool", 1 if val else 0

def _handle_int(val: Any) -> Tuple[str, int]:
    return "i64", int(val)

def _handle_float(val: Any) -> Tuple[str, float]:
    return "f64", float(val)

def _handle_str(val: str) -> Tuple[str, str]:
    return "str", val

def _handle_datetime(val: datetime) -> Tuple[str, str]:
    return "datetime", val.isoformat()

def _handle_bytes(val: bytes) -> Tuple[str, bytes]:
    return "bytes", val

def _handle_complex(val: complex) -> Tuple[str, bytes]:
    return "c128", np.array(val, dtype=np.complex128).tobytes()

_json_dumps = json.dumps

def _handle_json(val: Any) -> Tuple[str, str]:
    return "json", _json_dumps(val, default=_json_default)

_PRIMITIVE_HANDLERS = {
    bool: _handle_bool,
    int: _handle_int,
    float: _handle_float,
    str: _handle_str,
    datetime: _handle_datetime,
    bytes: _handle_bytes,
    complex: _handle_complex,
    dict: _handle_json,
    list: _handle_json,
}

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

def _make_ndarray_handler(dtype_str: str):
    def handler(val: np.ndarray):
        return dtype_str, val.tobytes()
    return handler

def _make_torch_handler(dtype_str: str):
    def handler(val: Any): # val is torch.Tensor
        return dtype_str, val.detach().cpu().numpy().tobytes()
    return handler

def get_or_make_handler(ch: Optional[ChannelDef], value: Any) -> Callable[[Any], Tuple[str, Any]]:
    # either 
    # 1. no channel go to first if statement
    # 2. serialized value and pickle
    # 3. serializer no pickle. then get value with recursive call to get handler to serialize the value with a basic handler.
    # 4. 
    # channel specfic handler of value.
    if ch is None:
        # if channel not specified, use old resolve_value logic
        v_type = type(value)
        handler = _PRIMITIVE_HANDLERS.get(v_type)
        if handler: return handler
        if isinstance(value, np.ndarray): return _make_ndarray_handler(_map_numpy_dtype(value.dtype))
        if torch and isinstance(value, torch.Tensor): return _make_torch_handler(_map_numpy_dtype(np.dtype(str(value.dtype).split('.')[-1])))
        return lambda v: resolve_value(None, v) # Fallback

    # If user has a serializer, we must run it first
    if ch.serializer:
        user_ser = ch.serializer
        if ch.pickle:
            _pickle_dumps = pickle.dumps
            def pickle_handler(val: Any):
                v = user_ser(val)
                return "json", {"json": _json_dumps(v, default=_json_default), "pickle": _pickle_dumps(v)}
            return pickle_handler
        
        # Non-pickle channel with serializer: we need to know what the serializer returns
        sample_out = user_ser(value)
        base_handler = get_or_make_handler(None, sample_out)
        def serialized_handler(val: Any):
            return base_handler(user_ser(val))
        return serialized_handler
    
    # channel exists but has no serializer, try to find a specialized handler based on the value type

    # primitives first
    v_type = type(value)
    handler = _PRIMITIVE_HANDLERS.get(v_type)
    if handler: return handler

    # numPy / torch
    if isinstance(value, np.ndarray):
        return _make_ndarray_handler(ch.dtype)
    if torch and isinstance(value, torch.Tensor):
        return _make_torch_handler(ch.dtype)

    # fallback to general resolve, rare case
    return lambda v: resolve_value(ch, v)

def resolve_value(channel_def: Optional[ChannelDef], value: Any) -> Tuple[str, Any]:
    # this remains for backward compatibility and auto detection
    if channel_def and channel_def.serializer is not None:
        value = channel_def.serializer(value)
        if channel_def.pickle:
            return "json", {"json": _json_dumps(value, default=_json_default), "pickle": pickle.dumps(value)}

    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        dtype_str = channel_def.dtype if channel_def else _map_numpy_dtype(value.dtype)
        return dtype_str, value.tobytes()

    # Fast check for primitives
    v_type = type(value)
    if v_type in _PRIMITIVE_HANDLERS:
        return _PRIMITIVE_HANDLERS[v_type](value)

    if isinstance(value, (int, np.integer)): return "i64", int(value)
    if isinstance(value, (float, np.floating)): return "f64", float(value)
    if isinstance(value, (dict, list)): return _handle_json(value)
    if isinstance(value, datetime): return _handle_datetime(value)

    raise TypeError(f"Cannot serialize {type(value).__name__}")

def _json_default(obj):
    if isinstance(obj, datetime): return obj.isoformat()
    if isinstance(obj, np.ndarray): return obj.tolist()
    if torch is not None and isinstance(obj, torch.Tensor): return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
