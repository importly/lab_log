import pytest
import requests
from lab_log import LabLogRun
import numpy as np

def test_reader_load(monkeypatch):
    run_id = "test_run"
    server = "localhost:1234"
    
    # Mock /runs/{id}/channels/{name}
    def mock_get(url, **kwargs):
        class MockResponse:
            def __init__(self, data):
                self.data = data
            def json(self):
                return self.data
            def raise_for_status(self):
                pass
        
        if "channels/sensor" in url:
            return MockResponse({
                "name": "sensor",
                "data": [1.0, 2.0, 3.0],
                "timestamps_ns": [100, 200, 300]
            })
        elif "channels" in url:
            return MockResponse({
                "channels": [{"name": "sensor"}]
            })
        elif url.endswith(run_id):
            return MockResponse({
                "status": "complete",
                "manifest": {"run_id": run_id}
            })
        return MockResponse({"error": "not found"})

    monkeypatch.setattr(requests, "get", mock_get)
    
    run = LabLogRun.open(server, run_id)
    assert run.status == "complete"
    assert run.list_channels() == ["sensor"]
    
    data = run.load("sensor")
    assert isinstance(data, np.ndarray)
    assert data.tolist() == [1.0, 2.0, 3.0]
    
    ts = run.load_timestamps("sensor")
    assert ts.tolist() == [100, 200, 300]
    
    ts_s = run.load_timestamps("sensor", unit="s")
    assert ts_s[0] == 100 / 1e9
