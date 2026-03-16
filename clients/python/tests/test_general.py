from pathlib import Path
from lab_log.logger import LabLog
import json

def test_general(tmp_path):
    cache_dir = tmp_path / "cache"
    
    experiment_id = "laser-testing"
    trial_name = "new wavelength test"
    
    logger = LabLog(
        trial_name=trial_name,
        experiment_id=experiment_id,
        researcher="aryan",
        hostname="lab-machine-01",
        cache_dir=str(cache_dir)
    )
    
    assert logger is not None
    
    assert logger.run_path.exists()
    assert logger.run_path.parent.name == experiment_id
    assert logger.run_path.name == logger.run_id
    
    manifest_file = logger.run_path / "manifest.json"
    assert manifest_file.exists()
    
    with open(manifest_file, "r") as f:
        manifest = json.load(f)
        assert manifest["trial_name"] == trial_name
        assert manifest["experiment_id"] == experiment_id
        assert manifest["run_id"] == logger.run_id
        assert "timestamp_start_utc" in manifest
        print(f"Verified manifest on disk: {manifest}")
