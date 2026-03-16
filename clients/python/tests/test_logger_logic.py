import pytest
import struct
import msgpack
import json
import numpy as np
from pathlib import Path
from lab_log.logger import LabLog

try:
    import torch
except ImportError:
    torch = None

def test_declare_channel_validation(tmp_path):
    cache_dir = tmp_path / "cache"
    logger = LabLog("test_trial", "test_exp", cache_dir=str(cache_dir))
    
    # Valid declaration (dtype from spec)
    logger.declare_channel("sensor", dtype="f64", unit="V")
    assert "sensor" in logger.channels
    assert logger.channels["sensor"].dtype == "f64"
    
    # Invalid dtype
    with pytest.raises(ValueError, match="Invalid dtype"):
        logger.declare_channel("bad", dtype="not_a_type")
        
    # Pickle constraints (Section 9)
    with pytest.raises(ValueError, match="pickle=True requires dtype='json'"):
        logger.declare_channel("bad_pickle", dtype="f64", pickle=True)
        
    with pytest.raises(ValueError, match="pickle=True requires dtype='json' and a custom serializer."):
        logger.declare_channel("bad_pickle", dtype="json", pickle=True)

def test_log_binary_format(tmp_path):
    cache_dir = tmp_path / "cache"
    logger = LabLog("test_trial", "test_exp", cache_dir=str(cache_dir))
    
    logger.declare_channel("temp", dtype="f64")
    test_val = 25.5
    logger.log("temp", test_val)
    
    log_bin = logger.run_path / "log.bin"
    assert log_bin.exists()
    
    with open(log_bin, "rb") as f:
        # Read 4-byte little-endian length prefix (Section 5)
        header = f.read(4)
        length = struct.unpack("<I", header)[0]
        
        # Read payload
        payload = f.read(length)
        data = msgpack.unpackb(payload)
        
        # Verify keys according to Section 5
        assert data["c"] == "temp"
        assert isinstance(data["t"], int) # timestamp in ns
        assert data["v"] == test_val

def test_numpy_serialization(tmp_path):
    cache_dir = tmp_path / "cache"
    logger = LabLog("test_trial", "test_exp", cache_dir=str(cache_dir))
    
    # Tier 1 Native
    logger.declare_channel("image", dtype="u8", shape=[2, 2])
    arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    logger.log("image", arr)
    
    log_bin = logger.run_path / "log.bin"
    with open(log_bin, "rb") as f:
        # Skip header
        f.read(4)
        payload = f.read()
        data = msgpack.unpackb(payload)
        
        # Tier 1 Native should be raw bytes in msgpack 'bin'
        assert isinstance(data["v"], bytes)
        assert data["v"] == arr.tobytes()

def test_adhoc_logging(tmp_path):
    cache_dir = tmp_path / "cache"
    logger = LabLog("test_trial", "test_exp", cache_dir=str(cache_dir))
    
    # Ad-hoc logging (no declaration)
    with pytest.warns(UserWarning, match="Ad-hoc logging"):
        logger.log("surprise", "hello")
        
    # Should still appear in manifest
    manifest = logger._generate_manifest()
    assert "surprise" in manifest["declared_channels"]

def test_context_manager(tmp_path):
    cache_dir = tmp_path / "cache"
    with LabLog("test", "test", cache_dir=str(cache_dir)) as logger:
        logger.log("test_ch", 1)
        # File should be open
        assert logger._log_file is not None
        assert not logger._log_file.closed
        
    # After context exit, file should be closed
    assert logger._log_file is None
