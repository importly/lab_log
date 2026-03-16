from lab_log.logger import LabLog

def test_general():
    logger = LabLog(
        trial_name="new wavelength test",
        experiment_id="laser-testing",
        researcher="aryan",
        hostname="lab-machine-01",
    )
    
    assert logger is not None
    
    runid = logger.run_id
    assert runid is not None
    print(f"Generated run ID: {runid}")
    
    manifest = logger._generate_manifest()
    assert manifest["trial_name"] == "new wavelength test"
    assert manifest["experiment_id"] == "laser-testing"
    assert manifest["run_id"] == runid
    assert "timestamp_start_utc" in manifest
    print(f"Generated manifest: {manifest}")
    
    
    
