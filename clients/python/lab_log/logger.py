import struct
import json
from datetime import datetime, timezone
from typing import Any, Optional
from pathlib import Path
import msgpack
import uuid
import base64
from importlib.metadata import version, PackageNotFoundError

class LabLog:
    trial_name: str
    experiment_id: str
    researcher: str
    hostname: str
    server: str
    run_id: str
    run_path: Path
    channels: dict
    config: dict
    
    timestamp_start: datetime # all times in utc
            
    def __init__(self, 
                 trial_name: str, 
                 experiment_id: str, 
                 researcher: str = "", 
                 hostname: str = "", 
                 server: str = "localhost:36524",
                 cache_dir: str = "~/.lablog/cache"):
        self.trial_name = trial_name
        self.experiment_id = experiment_id
        self.researcher = researcher
        self.hostname = hostname
        self.server = server
        self.run_id = self._generate_short_uuid()
        self.timestamp_start = datetime.now(timezone.utc)
        self.channels = {}
        
        self.config = {
            "sync_interval": 60,
            "chunk_size_mb": 10,
            "non_loggable_params": {},
            "cache_dir": cache_dir
        }

        # cache_dir/<experiment_id>/<run_id>/
        self.run_path = Path(self.config["cache_dir"]).expanduser() / self.experiment_id / self.run_id
        self.run_path.mkdir(parents=True, exist_ok=True)

        self._write_manifest() # save manifest

    def configure(self, 
                  sync_interval: Optional[int] = None, 
                  chunk_size_mb: Optional[int] = None, 
                  non_loggable_params: Optional[dict] = None, 
                  server: Optional[str] = None):
        
        if sync_interval is not None: self.config["sync_interval"] = sync_interval
        if chunk_size_mb is not None: self.config["chunk_size_mb"] = chunk_size_mb
        if non_loggable_params is not None: self.config["non_loggable_params"] = non_loggable_params
        if server is not None: self.server = server
        
        self._write_manifest()

    def _write_manifest(self):
        manifest_path = self.run_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self._generate_manifest(), f, indent=2)

    def declare_channel(self, channel_name: str, dtype: str, shape: Any = None, unit: str = ""):
        return
    
    def log(self, channel_name: str, data: Any):
        return
    
    def sync(self):
        return

    def complete(self, status: str = "complete"):
        return

    def _generate_short_uuid(self) -> str:
        u = uuid.uuid4()
        short = base64.urlsafe_b64encode(u.bytes).decode('utf-8')
        return short.rstrip('=')
    
    def _get_version(self) -> str:
        try:
            return version("lab-log")
        except PackageNotFoundError:
            return "0.0.0-dev"

    def _generate_manifest(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "trial_name": self.trial_name,
            "researcher": self.researcher,
            "hostname": self.hostname,
            "timestamp_start_utc": self.timestamp_start.isoformat(timespec='seconds').replace("+00:00", "Z"),
            "lab_log_version": self._get_version(),
            "server": self.server,
            "non_loggable_params": self.config["non_loggable_params"],
            "declared_channels": self.channels           
        }
