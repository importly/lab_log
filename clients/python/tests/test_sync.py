import pytest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
from lab_log.logger import LabLog
from lab_log.sync import ChunkSyncer


def test_chunk_upload_success(tmp_path):
    """Test that ChunkSyncer uploads a chunk and updates sync_state.json."""
    run_path = tmp_path / "run"
    run_path.mkdir()
    chunks_dir = run_path / "chunks"
    chunks_dir.mkdir()

    # Write a fake chunk file
    chunk_data = b"fake binary data for chunk 0"
    (chunks_dir / "0000.bin").write_bytes(chunk_data)

    # Write initial sync state
    (run_path / "sync_state.json").write_text(json.dumps({
        "confirmed_chunks": [],
        "pending_chunks": [],
        "total_frames_written": 0
    }))

    syncer = ChunkSyncer(run_path, "test-run-id", "localhost:36524")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"chunk_idx": 0, "checksum_sha256": "abc123"}

    with patch("requests.post", return_value=mock_response) as mock_post:
        result = syncer.upload_chunk(0)

    assert result is True

    # Verify the request was made correctly
    call_args = mock_post.call_args
    assert "/runs/test-run-id/chunks/0" in call_args[0][0]
    assert call_args[1]["data"] == chunk_data
    assert "X-Client-Upload-Ts-Ns" in call_args[1]["headers"]

    # Verify sync_state updated
    state = json.loads((run_path / "sync_state.json").read_text())
    assert 0 in state["confirmed_chunks"]


def test_sync_pending_uploads_all(tmp_path):
    """Test that sync_pending uploads all unconfirmed chunks."""
    run_path = tmp_path / "run"
    run_path.mkdir()
    chunks_dir = run_path / "chunks"
    chunks_dir.mkdir()

    for i in range(3):
        (chunks_dir / f"{i:04d}.bin").write_bytes(f"chunk {i} data".encode())

    (run_path / "sync_state.json").write_text(json.dumps({
        "confirmed_chunks": [0],  # 0 already done
        "pending_chunks": [],
        "total_frames_written": 0
    }))

    syncer = ChunkSyncer(run_path, "test-run-id", "localhost:36524")

    def mock_post_fn(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        idx = int(url.split("/")[-1])
        resp.json.return_value = {"chunk_idx": idx, "checksum_sha256": f"hash{idx}"}
        return resp

    with patch("requests.post", side_effect=mock_post_fn):
        count = syncer.sync_pending()

    assert count == 2  # only chunks 1 and 2 were new


def test_upload_failure_marks_pending(tmp_path):
    """Test that a failed upload adds the chunk to pending_chunks."""
    run_path = tmp_path / "run"
    run_path.mkdir()
    chunks_dir = run_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "0000.bin").write_bytes(b"data")
    (run_path / "sync_state.json").write_text(json.dumps({
        "confirmed_chunks": [], "pending_chunks": [], "total_frames_written": 0
    }))

    syncer = ChunkSyncer(run_path, "test-run-id", "localhost:36524")

    with patch("requests.post", side_effect=Exception("connection refused")):
        result = syncer.upload_chunk(0)

    assert result is False
    state = json.loads((run_path / "sync_state.json").read_text())
    assert 0 in state["pending_chunks"]
    assert 0 not in state["confirmed_chunks"]


def test_upload_non_200_marks_pending(tmp_path):
    """Test that a non-200 response adds the chunk to pending_chunks."""
    run_path = tmp_path / "run"
    run_path.mkdir()
    chunks_dir = run_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "0000.bin").write_bytes(b"data")
    (run_path / "sync_state.json").write_text(json.dumps({
        "confirmed_chunks": [], "pending_chunks": [], "total_frames_written": 0
    }))

    syncer = ChunkSyncer(run_path, "test-run-id", "localhost:36524")

    mock_response = MagicMock()
    mock_response.status_code = 500

    with patch("requests.post", return_value=mock_response):
        result = syncer.upload_chunk(0)

    assert result is False
    state = json.loads((run_path / "sync_state.json").read_text())
    assert 0 in state["pending_chunks"]
    assert 0 not in state["confirmed_chunks"]


def test_sync_pending_skips_confirmed(tmp_path):
    """Test that sync_pending doesn't re-upload already confirmed chunks."""
    run_path = tmp_path / "run"
    run_path.mkdir()
    chunks_dir = run_path / "chunks"
    chunks_dir.mkdir()

    for i in range(3):
        (chunks_dir / f"{i:04d}.bin").write_bytes(f"chunk {i} data".encode())

    # All chunks already confirmed
    (run_path / "sync_state.json").write_text(json.dumps({
        "confirmed_chunks": [0, 1, 2],
        "pending_chunks": [],
        "total_frames_written": 0
    }))

    syncer = ChunkSyncer(run_path, "test-run-id", "localhost:36524")

    with patch("requests.post") as mock_post:
        count = syncer.sync_pending()

    assert count == 0
    mock_post.assert_not_called()


def test_complete_posts_to_server(tmp_path):
    """Test that complete() POSTs to the correct URL with correct body."""
    run_path = tmp_path / "run"
    run_path.mkdir()

    syncer = ChunkSyncer(run_path, "test-run-id", "localhost:36524")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"h5_checksum_sha256": "deadbeef", "status": "complete"}

    with patch("requests.post", return_value=mock_response) as mock_post:
        result = syncer.complete(final_chunk_idx=5)

    assert result == {"h5_checksum_sha256": "deadbeef", "status": "complete"}
    call_args = mock_post.call_args
    assert "/runs/test-run-id/complete" in call_args[0][0]
    assert call_args[1]["json"] == {"final_chunk_idx": 5}


def test_complete_returns_none_on_failure(tmp_path):
    """Test that complete() returns None when the server call fails."""
    run_path = tmp_path / "run"
    run_path.mkdir()

    syncer = ChunkSyncer(run_path, "test-run-id", "localhost:36524")

    with patch("requests.post", side_effect=Exception("timeout")):
        result = syncer.complete(final_chunk_idx=3)

    assert result is None


def test_load_state_default_when_missing(tmp_path):
    """Test that _load_state returns defaults when sync_state.json doesn't exist."""
    run_path = tmp_path / "run"
    run_path.mkdir()

    syncer = ChunkSyncer(run_path, "test-run-id", "localhost:36524")
    state = syncer._load_state()

    assert state["confirmed_chunks"] == []
    assert state["pending_chunks"] == []
    assert state["total_frames_written"] == 0


def test_upload_chunk_missing_file(tmp_path):
    """Test that upload_chunk returns False when chunk file doesn't exist."""
    run_path = tmp_path / "run"
    run_path.mkdir()
    (run_path / "chunks").mkdir()
    (run_path / "sync_state.json").write_text(json.dumps({
        "confirmed_chunks": [], "pending_chunks": [], "total_frames_written": 0
    }))

    syncer = ChunkSyncer(run_path, "test-run-id", "localhost:36524")

    with patch("requests.post") as mock_post:
        result = syncer.upload_chunk(99)

    assert result is False
    mock_post.assert_not_called()
