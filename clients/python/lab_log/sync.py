import json
import time
import threading
import warnings
from pathlib import Path
from typing import Optional, Callable
import requests


class ChunkSyncer:
    def __init__(self, run_path: Path, run_id: str, server: str, ssl_verify: bool | str = True):
        self.run_path = run_path
        self.run_id = run_id
        self.server = server
        self.ssl_verify = ssl_verify
        self._state_path = run_path / "sync_state.json" # where chunks are confirmed to be
        self._lock = threading.Lock()
        self._uploading_indices = set()

    def _load_state(self) -> dict:
        if not self._state_path.exists():
            return {
                "confirmed_chunks": [],
                "pending_chunks": [],
                "total_frames_written": 0,
            }
        try:
            with open(self._state_path, "r") as f:
                return json.load(f)
        except Exception:
            return {
                "confirmed_chunks": [],
                "pending_chunks": [],
                "total_frames_written": 0,
            }

    def _save_state(self, state: dict):
        tmp_path = self._state_path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(state, f, indent=2)
            tmp_path.replace(self._state_path)
        except Exception as e:
            warnings.warn(f"Failed to save sync state: {e}")
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def upload_chunk(self, chunk_idx: int) -> bool:
        with self._lock:
            if chunk_idx in self._uploading_indices:
                return False
            self._uploading_indices.add(chunk_idx)

        try:
            return self._do_upload_chunk(chunk_idx)
        finally:
            with self._lock:
                self._uploading_indices.remove(chunk_idx)

    def _do_upload_chunk(self, chunk_idx: int) -> bool:
        chunk_path = self.run_path / "chunks" / f"{chunk_idx:04d}.bin"
        if not chunk_path.exists():
            warnings.warn(f"Chunk file not found: {chunk_path}")
            return False

        try:
            bytes_content = open(chunk_path, "rb").read()
        except Exception as e:
            warnings.warn(f"Failed to read chunk {chunk_idx}: {e}")
            return False

        url = f"https://{self.server}/runs/{self.run_id}/chunks/{chunk_idx}"
        try:
            resp = requests.post(
                url,
                data=bytes_content,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Client-Upload-Ts-Ns": str(time.time_ns()),
                },
                timeout=(2.0, 30.0),  # 2s connect, 30s read
                verify=self.ssl_verify,
            )
        except Exception as e:
            warnings.warn(f"Failed to upload chunk {chunk_idx}: {e}")
            with self._lock:
                state = self._load_state()
                if chunk_idx not in state["pending_chunks"]:
                    state["pending_chunks"].append(chunk_idx)
                self._save_state(state)
            return False

        if resp.status_code == 200:
            with self._lock:
                state = self._load_state()
                if chunk_idx not in state["confirmed_chunks"]:
                    state["confirmed_chunks"].append(chunk_idx)
                if chunk_idx in state["pending_chunks"]:
                    state["pending_chunks"].remove(chunk_idx)
                self._save_state(state)
            return True
        else:
            warnings.warn(f"Server returned {resp.status_code} for chunk {chunk_idx}: {resp.text}")
            with self._lock:
                state = self._load_state()
                if chunk_idx not in state["pending_chunks"]:
                    state["pending_chunks"].append(chunk_idx)
                self._save_state(state)
            return False

    def sync_pending(self) -> int:
        chunks_dir = self.run_path / "chunks"
        if not chunks_dir.exists():
            return 0

        with self._lock:
            state = self._load_state()
        
        confirmed = set(state["confirmed_chunks"])

        chunk_files = sorted(
            chunks_dir.glob("*.bin"),
            key=lambda p: int(p.stem),
        )

        newly_confirmed = 0
        for chunk_file in chunk_files:
            try:
                chunk_idx = int(chunk_file.stem)
            except ValueError:
                continue

            if chunk_idx in confirmed:
                continue

            if self.upload_chunk(chunk_idx):
                newly_confirmed += 1

        return newly_confirmed

    def complete(self, final_chunk_idx: int) -> Optional[dict]:
        url = f"https://{self.server}/runs/{self.run_id}/complete"
        try:
            resp = requests.post(
                url,
                json={"final_chunk_idx": final_chunk_idx},
                timeout=(2.0, 30.0),  # 2s connect, 30s read
                verify=self.ssl_verify,
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                warnings.warn(f"Server returned {resp.status_code} on complete")
                return None
        except Exception as e:
            warnings.warn(f"Failed to complete run on server: {e}")
            return None


class BackgroundSyncThread:
    def __init__(self, sync_fn: Callable, interval_seconds: int = 30):
        self._sync_fn = sync_fn
        self._interval = interval_seconds
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="lab-log-bg-sync")
        self._thread.start()

    def _run(self):
        while not self._stop_event.wait(timeout=self._interval):
            try:
                self._sync_fn()
            except Exception as e:
                warnings.warn(f"Background sync error: {e}")

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
