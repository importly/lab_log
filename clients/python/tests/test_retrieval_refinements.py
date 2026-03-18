import pytest
import numpy as np
import base64
import pickle
import json
from unittest.mock import MagicMock, patch
from lab_log.reader import LabLogRun
from lab_log.logger import LabLog

def test_load_timestamps_unit_s():
    """Test LabLogRun.load_timestamps(..., unit='s') returns seconds."""
    run = LabLogRun("localhost:36524", "test_run")
    
    # Mock response from server
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "timestamps_ns": [1000000000, 2000000000, 3000000000]
    }
    mock_resp.status_code = 200
    
    with patch("requests.get", return_value=mock_resp):
        ts = run.load_timestamps("ch", unit="s")
        assert isinstance(ts, np.ndarray)
        assert np.array_equal(ts, np.array([1.0, 2.0, 3.0]))

def test_load_as_torch():
    """Test LabLogRun.load(..., as_torch=True) returns a torch Tensor."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
        
    run = LabLogRun("localhost:36524", "test_run")
    
    # Mock response from server
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [1.0, 2.0, 3.0],
        "timestamps_ns": [1, 2, 3]
    }
    mock_resp.status_code = 200
    
    with patch("requests.get", return_value=mock_resp):
        data = run.load("ch", as_torch=True)
        assert torch.is_tensor(data)
        assert torch.equal(data, torch.tensor([1.0, 2.0, 3.0]))

def test_export_torch():
    """Test LabLogRun.export(..., format='torch') requests correct format."""
    run = LabLogRun("localhost:36524", "test_run")
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_content.return_value = [b"mock_torch_data"]
    
    with patch("requests.get", return_value=mock_resp) as mock_get:
        # Use a temporary file path
        run.export("ch", "test.pt", format="torch")
        
        # Verify URL called has format=torch
        args, kwargs = mock_get.call_args
        assert "format=torch" in args[0]

class MyCustomClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

def test_tier3_pickle_retrieval():
    """Test Tier 3 (Pickle): reconstruction of objects via LabLogRun.load()."""
    obj = MyCustomClass(10, 20)
    p_bytes = pickle.dumps(obj)
    p_b64 = base64.b64encode(p_bytes).decode('utf-8')
    
    run = LabLogRun("localhost:36524", "test_run")
    
    # Mock response representing a pickle-enabled channel
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "is_pickle": True,
        "data": [json.dumps({"x": 10, "y": 20})],
        "pickle": [p_b64],
        "timestamps_ns": [123456789]
    }
    mock_resp.status_code = 200
    
    with patch("requests.get", return_value=mock_resp):
        retrieved = run.load("ch")
        assert isinstance(retrieved, list)
        assert len(retrieved) == 1
        assert retrieved[0] == obj
        assert isinstance(retrieved[0], MyCustomClass)

def test_adhoc_logging_no_pickle(tmp_path):
    """Verify that ad-hoc logging doesn't use pickle even if it's a complex object."""
    cache_dir = tmp_path / "cache"
    class MyData:
        def __init__(self, x):
            self.x = x
            
    with LabLog("adhoc_test", "adhoc_exp", cache_dir=str(cache_dir)) as logger:
        # Ad-hoc log a dict (which resolve_value will handle as json)
        logger.log("adhoc_ch", {"val": 42})
        
        assert "adhoc_ch" in logger.channels
        assert logger.channels["adhoc_ch"].pickle is False
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not installed")

    import sqlite3
    try:
        from server.src.assemble import assemble_hdf5
    except (ImportError, ModuleNotFoundError):
        pytest.skip("server module not found")
    
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE runs (run_id TEXT, experiment_id TEXT, trial_name TEXT, trial_number INTEGER, researcher TEXT, hostname TEXT, timestamp_start TEXT, status TEXT, manifest_json TEXT)")
    conn.execute("CREATE TABLE chunks (run_id TEXT, chunk_idx INTEGER, received_at TEXT, clock_offset_ns INTEGER)")
    
    run_id = "test_run"
    manifest = {
        "declared_channels": [
            {"name": "ch1", "dtype": "f32"}
        ]
    }
    conn.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                 (run_id, "test_exp", "trial", 1, "me", "host", "2023-01-01T00:00:00Z", "complete", json.dumps(manifest)))
    
    # Create a mock chunk file
    storage_root = tmp_path / "storage"
    run_dir = storage_root / "test_exp" / run_id
    (run_dir / "chunks").mkdir(parents=True)
    
    import msgpack
    import struct
    
    chunk_file = run_dir / "chunks" / "0000.bin"
    with open(chunk_file, "wb") as f:
        # Frame: {c: "ch1", t: 1000, v: b'\x00\x00\x80?'} (1.0 in f32)
        payload = {"c": "ch1", "t": 1000, "v": b'\x00\x00\x80?'}
        encoded = msgpack.packb(payload)
        f.write(struct.pack("<I", len(encoded)))
        f.write(encoded)
        
    conn.execute("INSERT INTO chunks VALUES (?, ?, ?, ?)", (run_id, 0, "2023-01-01 00:01:00", 500))
    conn.commit()
    conn.close()
    
    assemble_hdf5(str(db_path), run_id, str(storage_root))
    
    h5_path = run_dir / f"{run_id}.h5"
    assert h5_path.exists()
    
    with h5py.File(h5_path, 'r') as f:
        ch1 = f['declared/ch1']
        assert "timestamps_ns" in ch1
        assert "timestamps_raw_ns" in ch1
        
        # timestamps_ns should have offset: 1000 + 500 = 1500
        assert ch1['timestamps_ns'][0] == 1500
        # timestamps_raw_ns should be raw: 1000
        assert ch1['timestamps_raw_ns'][0] == 1000
