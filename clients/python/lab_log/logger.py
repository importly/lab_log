import struct
from datetime import datetime, timezone
import msgpack
import uuid
import base64
from importlib.metadata import version

class LabLog:
    trial_name:str
    experiment_id:str
    researcher:str
    hostname:str
    server:str
    channels:dict
    config:dict = {
        "sync_interval": 60,
        "chunk_size_mb": 10,
        "non_loggable_params": {},
        "cache_dir": "./.lablog/cache"
    }
    run_id:str
    
    timestamp_start:datetime # all times in utc
    
        
    
    def __init__(self, trial_name:str, experiment_id:str, researcher:str = "", hostname:str = "", server:str = "localhost:36524"):
        self.trial_name = trial_name
        self.experiment_id = experiment_id
        self.researcher = researcher
        self.hostname = hostname
        self.run_id = self._generate_short_uuid()
        self.timestamp_start = datetime.now(timezone.utc)
        self.server = server
        
        self.channels = {}
        return
    
    def configure(self, sync_interval:int, chunk_size_mb:int,non_loggable_params:dict, cashe_dire:str = "./.lablog/cache"):
        return
    
    def declare_channel(self,channel_name:str,dtype:str,shape,unit):
        return
    
    def log(self,channel_name:str,data):
        return
    
    def sync(self):
        return
    def complete(self, status:str="complete"):
        return
    

    def _generate_short_uuid(self):
        u = uuid.uuid4()
        short = base64.urlsafe_b64encode(u.bytes).decode('utf-8')
        return short.rstrip('=')
    
    def _generate_manifest(self):
        return {
            "trial_name": self.trial_name,
            "experiment_id": self.experiment_id,
            "researcher": self.researcher,
            "hostname": self.hostname,
            "timestamp_start_utc": self.timestamp_start.isoformat(timespec='seconds').replace("+00:00", "Z"),
            "server": self.server,
            "run_id": self.run_id,
            "lab_log_version": version("lab-log"),
            "non_loggable_params": self.config["non_loggable_params"],
            "declared_channels": self.channels           
        }