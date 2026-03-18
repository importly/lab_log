import requests
import numpy as np
import json
import base64
import pickle
from typing import Optional, List, Dict, Any, Union

# inital work on reader, not sure if it works, need to work on pickle system, may move to https://github.com/cloudpipe/cloudpickle
class LabLogRun:
    def __init__(self, server: str, run_id: str, ssl_verify: bool | str = True):
        self.server = server
        self.run_id = run_id
        self.ssl_verify = ssl_verify
        self._manifest = None
        self._status = None

    @classmethod
    def open(cls, server: str, run_id: str, ssl_verify: bool | str = True) -> "LabLogRun":
        return cls(server, run_id, ssl_verify=ssl_verify)

    def _fetch_metadata(self):
        if self._manifest is not None:
            return
        url = f"https://{self.server}/runs/{self.run_id}"
        resp = requests.get(url, verify=self.ssl_verify)
        resp.raise_for_status()
        data = resp.json()
        self._manifest = data.get("manifest", {})
        self._status = data.get("status")

    @property
    def status(self) -> Optional[str]:
        self._fetch_metadata()
        return self._status

    def get_manifest(self) -> Dict[str, Any]:
        self._fetch_metadata()
        return self._manifest or {}

    def list_channels(self) -> List[str]:
        url = f"https://{self.server}/runs/{self.run_id}/channels"
        resp = requests.get(url, verify=self.ssl_verify)
        resp.raise_for_status()
        return [ch["name"] for ch in resp.json().get("channels", [])]

    def load(self, channel_name: str, index: Optional[Union[int, tuple, slice]] = None, as_torch: bool = False, as_json: bool = False) -> Any:
        params = {}
        if index is not None:
            if isinstance(index, int):
                params["start"] = index
                params["end"] = index + 1
            elif isinstance(index, tuple) and len(index) == 2:
                params["start"] = index[0]
                params["end"] = index[1]
            elif isinstance(index, slice):
                params["start"] = index.start if index.start is not None else 0
                params["end"] = index.stop if index.stop is not None else -1

        url = f"https://{self.server}/runs/{self.run_id}/channels/{channel_name}"
        resp = requests.get(url, params=params, verify=self.ssl_verify)
        resp.raise_for_status()
        data = resp.json()
        
        if "error" in data:
            raise ValueError(data["error"])
            
        is_pickle = data.get("is_pickle", False)
        values = data["data"]
        pickles = data.get("pickle")

        if as_json:
            results = [json.loads(v) if isinstance(v, str) else v for v in values]
            if isinstance(index, int):
                return results[0] if results else None
            return results

        if is_pickle:
            # reconstruct objects from pickle
            results = []
            if pickles:
                for p_b64 in pickles:
                    if p_b64:
                        results.append(pickle.loads(base64.b64decode(p_b64)))
                    else:
                        results.append(None)
            
            if isinstance(index, int):
                return results[0] if results else None
            return results

        # standard data
        arr = np.array(values)
        
        if isinstance(index, int) and arr.ndim > 0:
            arr = arr[0]

        if as_torch:
            import torch
            return torch.from_numpy(arr)
        return arr

    def load_timestamps(self, channel_name: str, index: Optional[Union[int, tuple, slice]] = None, unit: str = "ns") -> np.ndarray:
        params = {}
        if index is not None:
            if isinstance(index, int):
                params["start"] = index
                params["end"] = index + 1
            elif isinstance(index, tuple) and len(index) == 2:
                params["start"] = index[0]
                params["end"] = index[1]
            elif isinstance(index, slice):
                params["start"] = index.start if index.start is not None else 0
                params["end"] = index.stop if index.stop is not None else -1

        url = f"https://{self.server}/runs/{self.run_id}/channels/{channel_name}"
        resp = requests.get(url, params=params, verify=self.ssl_verify)
        resp.raise_for_status()
        data = resp.json()
        
        ts = np.array(data["timestamps_ns"], dtype=np.int64)
        if isinstance(index, int) and ts.ndim > 0:
            ts = ts[0]
            
        if unit == "s":
            return ts / 1e9
        return ts

    def export(self, channel_name: str, path: str, format: str = "npy", as_json: bool = False):
        fmt = "json" if as_json else format
        if fmt == "pt": fmt = "torch"
        
        url = f"https://{self.server}/runs/{self.run_id}/export/{channel_name}?format={fmt}"
        resp = requests.get(url, stream=True, verify=self.ssl_verify)
        resp.raise_for_status()
        
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
