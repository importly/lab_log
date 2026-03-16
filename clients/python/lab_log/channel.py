from dataclasses import dataclass, field
from typing import Any, Callable, Optional, List

# Valid dtypes, will define in final documentation
VALID_DTYPES = {
    "f16", "f32", "f64",
    "i8", "i16", "i32", "i64",
    "u8", "u16", "u32", "u64",
    "c64", "c128",
    "bool", "str", "json", "bytes", "datetime"
}

@dataclass
class ChannelDef:
    name: str
    dtype: str
    shape: Optional[List[Optional[int]]] = None
    unit: Optional[str] = None
    description: Optional[str] = None
    compression: str = "gzip"
    compression_level: int = 4
    frequency_hz: Optional[float] = None
    serializer: Optional[Callable[[Any], Any]] = None
    pickle: bool = False # pickle saves the Python object directly and has json for fallback
    _cached_handler: Optional[Callable[[Any], Any]] = field(default=None, repr=False, compare=False) # cached serialization handler

    def validate(self):
        if self.dtype not in VALID_DTYPES:
            raise ValueError(f"Invalid dtype '{self.dtype}'. Must be one of {VALID_DTYPES}")
        
        if self.pickle:
            if self.dtype != "json" or self.serializer is None:
                raise ValueError("pickle=True requires dtype='json' and a custom serializer.")
        
        if self.shape is not None and not isinstance(self.shape, (list, tuple)):
            raise ValueError("shape must be a list of integers or None.")

    def to_dict(self) -> dict:
        """Returns a dict representation for the manifest, excluding local-only fields."""
        d = {
            "name": self.name,
            "dtype": self.dtype,
            "shape": self.shape,
            "unit": self.unit,
            "compression": self.compression,
            "compression_level": self.compression_level,
        }
        if self.frequency_hz:
            d["frequency_hz"] = self.frequency_hz
        return d
