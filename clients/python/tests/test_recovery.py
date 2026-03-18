import pytest
import shutil
import json
from pathlib import Path
from lab_log import LabLog

def test_find_orphaned_runs(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    
    # Configure global cache dir for test
    LabLog.configure_global(cache_dir=str(cache_dir))
    
    # Create an orphaned run manually
    exp_id = "test_exp"
    run_id = "orphan_run_1"
    run_path = cache_dir / exp_id / run_id
    run_path.mkdir(parents=True)
    
    manifest = {
        "run_id": run_id,
        "experiment_id": exp_id,
        "trial_name": "orphan_trial"
    }
    with open(run_path / "manifest.json", "w") as f:
        json.dump(manifest, f)
        
    # It should be found as an orphan (no .active file)
    orphans = LabLog.find_orphaned_runs()
    assert len(orphans) == 1
    assert orphans[0]["run_id"] == run_id
    
    # Now start a real logger
    logger = LabLog("new_trial", "new_exp")
    
    # The active run should NOT be found as an orphan
    orphans = LabLog.find_orphaned_runs()
    assert len(orphans) == 1 # still only the old one
    assert all(o["run_id"] != logger.run_id for o in orphans)
    
    logger.complete()

def test_discard_orphan(tmp_path):
    cache_dir = tmp_path / "cache"
    LabLog.configure_global(cache_dir=str(cache_dir))
    
    exp_id = "test_exp"
    run_id = "orphan_run_2"
    run_path = cache_dir / exp_id / run_id
    run_path.mkdir(parents=True)
    with open(run_path / "manifest.json", "w") as f:
        json.dump({"run_id": run_id, "experiment_id": exp_id}, f)
        
    assert len(LabLog.find_orphaned_runs()) == 1
    LabLog.discard_orphan(run_id)
    assert len(LabLog.find_orphaned_runs()) == 0
    assert not run_path.exists()

def test_auto_recover_on_init(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    LabLog.configure_global(cache_dir=str(cache_dir), auto_recover_orphans=True)
    
    exp_id = "test_exp"
    run_id = "orphan_run_3"
    run_path = cache_dir / exp_id / run_id
    run_path.mkdir(parents=True)
    with open(run_path / "manifest.json", "w") as f:
        json.dump({"run_id": run_id, "experiment_id": exp_id}, f)
        
    # Mock finalize_orphan to avoid network calls
    finalized = []
    def mock_finalize(rid, server, **kwargs):
        finalized.append(rid)
    monkeypatch.setattr(LabLog, "finalize_orphan", mock_finalize)
    
    logger = LabLog("trial", "exp")
    assert run_id in finalized
    logger.complete()

def test_cache_cleanup_on_complete(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    LabLog.configure_global(cache_dir=str(cache_dir))
    
    # Mock syncer to return success
    from lab_log.sync import ChunkSyncer
    def mock_complete(self, final_idx):
        return {"status": "complete"}
    monkeypatch.setattr(ChunkSyncer, "complete", mock_complete)
    
    logger = LabLog("trial", "exp")
    run_path = logger.run_path
    assert run_path.exists()
    
    logger.complete()
    assert not run_path.exists()
