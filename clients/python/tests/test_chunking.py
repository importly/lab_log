import pytest
import os
from pathlib import Path
from lab_log.logger import LabLog

def test_chunk_rotation(tmp_path):
    # Set a very small chunk size to trigger rotation
    cache_dir = tmp_path / "cache"
    
    # 0.001 MB = 1 KB
    with LabLog("test", "test", cache_dir=str(cache_dir)) as logger:
        logger.configure(chunk_size_mb=0.001)
        
        # Log enough data to trigger rotation multiple times
        # Each 'int' log is ~30-40 bytes
        for i in range(100):
            logger.log("ch", i)
            
        # Verify chunks folder has files
        chunks_dir = logger.run_path / "chunks"
        chunk_files = list(chunks_dir.glob("*.bin"))
        
        # We expect at least one rotation
        assert len(chunk_files) > 0
        # Check naming convention
        assert (chunks_dir / "0000.bin").exists()
        
        # Check logger state
        assert logger._log_file_count > 0
