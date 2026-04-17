"""
Cloud Run Job service — handles:
  - Streaming audio upload to GCS while recording
  - Submitting a Cloud Run Job execution
  - Polling execution status
  - Downloading the transcript result from GCS
  - Cleaning up GCS session folder
"""
from __future__ import annotations

import io
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import os

import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class JobConfig:
    session_id: str
    session_folder_uri: str   # gs://bucket/{session_id}/
    language: str = "auto"
    output_language: str = "vi"   # vi hoặc ja
    initial_prompt: str = ""


@dataclass
class JobExecutionStatus:
    execution_name: str       # projects/.../jobs/.../executions/{id}
    done: bool
    succeeded: bool
    error: str | None


# ---------------------------------------------------------------------------
# Streaming upload helper
# ---------------------------------------------------------------------------

class GCSStreamingUpload:
    """
    Wraps a resumable GCS upload so audio chunks can be written incrementally
    during recording, then finalised when recording stops.

    Usage:
        upload = cloud_run_service.start_streaming_upload(session_id)
        upload.write(chunk_bytes)   # called per 100 ms audio chunk
        upload.finalize()           # called after stop_recording
    """

    # GCS resumable upload requires multiples of 256 KiB (except the last chunk)
    _CHUNK_SIZE = 256 * 1024

    # Minimal streaming WAV header: uses 0xFFFFFFFF as placeholder sizes.
    # soundfile / librosa / ffmpeg all handle this gracefully.
    _SAMPLE_RATE = 16000
    _CHANNELS = 1
    _BITS = 16

    @staticmethod
    def _wav_header() -> bytes:
        """44-byte PCM WAV header with placeholder data/file sizes for streaming."""
        import struct
        num_channels = GCSStreamingUpload._CHANNELS
        sample_rate  = GCSStreamingUpload._SAMPLE_RATE
        bits         = GCSStreamingUpload._BITS
        byte_rate    = sample_rate * num_channels * bits // 8
        block_align  = num_channels * bits // 8
        placeholder  = 0xFFFFFFFF  # unknown size — standard streaming WAV practice
        return struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", placeholder,      # ChunkID, ChunkSize (unknown)
            b"WAVE",
            b"fmt ", 16,               # Subchunk1ID, Subchunk1Size (PCM = 16)
            1,                         # AudioFormat (PCM)
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits,
            b"data", placeholder,      # Subchunk2ID, Subchunk2Size (unknown)
        )

    def __init__(self, bucket_name: str, object_name: str, credentials) -> None:
        from google.cloud import storage as gcs
        self._client = gcs.Client(credentials=credentials)
        self._bucket = self._client.bucket(bucket_name)
        self._blob = self._bucket.blob(object_name)
        self._lock = threading.Lock()
        self._done = False
        self._error: Exception | None = None
        # Open resumable upload and immediately write WAV header
        self._upload = self._blob.open("wb", chunk_size=self._CHUNK_SIZE)
        self._upload.write(self._wav_header())

    def write(self, data: bytes) -> None:
        """Write a chunk of raw PCM int16 bytes. Thread-safe."""
        if self._done or self._error:
            return
        try:
            with self._lock:
                self._upload.write(data)
        except Exception as exc:
            self._error = exc

    def finalize(self) -> None:
        """Close the upload stream, flushing all remaining buffered data to GCS."""
        if self._done:
            return
        if self._error:
            raise self._error
        with self._lock:
            self._upload.close()
        self._done = True

    @property
    def error(self) -> Exception | None:
        return self._error


# ---------------------------------------------------------------------------
# Cloud Run Service
# ---------------------------------------------------------------------------

_CLOUD_RUN_BASE = "https://run.googleapis.com/v2"
_GCS_BASE = "https://storage.googleapis.com/storage/v1"

_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
]


class CloudRunService:
    def __init__(self) -> None:
        self.project_id = os.environ.get("GCP_PROJECT_ID", "")
        self.region     = os.environ.get("GCR_REGION", "asia-east1")
        self.job_name   = os.environ.get("GCR_JOB_NAME", "whisper-transcriber")
        self.gcs_bucket = os.environ.get("GCS_BUCKET", "")
        self._credentials = None
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if creds_path:
            self._load_credentials(creds_path)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_credentials(self, credentials_path: str) -> None:
        self._credentials_path = credentials_path
        self._load_credentials(credentials_path)

    def set_bucket(self, bucket_name: str) -> None:
        self.gcs_bucket = bucket_name

    def _load_credentials(self, path: str) -> None:
        self._credentials = service_account.Credentials.from_service_account_file(
            path, scopes=_SCOPES,
        )

    def _get_credentials(self):
        if self._credentials is None:
            raise RuntimeError(
                "Google Cloud credentials not configured. "
                "Set credentials_path before using CloudRunService."
            )
        # Refresh if expired
        if not self._credentials.valid:
            self._credentials.refresh(Request())
        return self._credentials

    def _auth_header(self) -> dict[str, str]:
        creds = self._get_credentials()
        return {"Authorization": f"Bearer {creds.token}"}

    # ------------------------------------------------------------------
    # Streaming upload
    # ------------------------------------------------------------------

    def start_streaming_upload(self, session_id: str) -> GCSStreamingUpload:
        """
        Opens a resumable GCS upload for gs://bucket/{session_id}/audio.wav.
        Call .write(chunk) with each audio chunk, then .finalize() when done.
        """
        object_name = f"{session_id}/audio.wav"
        return GCSStreamingUpload(
            bucket_name=self.gcs_bucket,
            object_name=object_name,
            credentials=self._get_credentials(),
        )

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    def submit_job(self, config: JobConfig) -> str:
        """
        Trigger a Cloud Run Job execution with env var overrides.
        Returns the execution resource name (persistent identifier).
        """
        url = (
            f"{_CLOUD_RUN_BASE}/projects/{self.project_id}"
            f"/locations/{self.region}/jobs/{self.job_name}:run"
        )
        body = {
            "overrides": {
                "containerOverrides": [
                    {
                        "env": [
                            {"name": "SESSION_FOLDER_URI", "value": config.session_folder_uri},
                            {"name": "SESSION_ID",         "value": config.session_id},
                            {"name": "LANGUAGE",           "value": config.language},
                            {"name": "OUTPUT_LANGUAGE",    "value": config.output_language},
                            {"name": "INITIAL_PROMPT",     "value": config.initial_prompt},
                        ]
                    }
                ]
            }
        }
        resp = requests.post(url, json=body, headers=self._auth_header(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        print(f"[cloud_run] submit response: {json.dumps(data, indent=2)}")

        # The :run endpoint returns a google.longrunning.Operation.
        # The execution resource name lives in metadata.name.
        # Fall back to extracting from operation name if needed:
        #   operations/projects/.../locations/.../jobs/.../executions/EXEC_ID/operations/...
        execution_name = data.get("metadata", {}).get("name", "")
        if not execution_name:
            # Try to derive from operation name
            op_name = data.get("name", "")
            # op_name pattern: .../executions/{id}/operations/{op}
            if "/executions/" in op_name:
                execution_name = op_name.split("/operations/")[0]
        print(f"[cloud_run] execution_name: {execution_name}")
        return execution_name

    # ------------------------------------------------------------------
    # Execution polling
    # ------------------------------------------------------------------

    def poll_execution(self, execution_name: str) -> JobExecutionStatus:
        """
        Fetch the current status of a Cloud Run Job execution.
        execution_name: full resource name from submit_job()
        """
        url = f"{_CLOUD_RUN_BASE}/{execution_name}"
        resp = requests.get(url, headers=self._auth_header(), timeout=30)
        resp.raise_for_status()
        data = resp.json()

        conditions: list[dict] = data.get("conditions", [])
        print(f"[cloud_run] poll conditions: {conditions}")

        completed = next((c for c in conditions if c.get("type") == "Completed"), None)

        # state values: CONDITION_PENDING | CONDITION_RECONCILING | CONDITION_SUCCEEDED | CONDITION_FAILED
        state = (completed or {}).get("state", "CONDITION_PENDING")

        if state in ("CONDITION_PENDING", "CONDITION_RECONCILING"):
            return JobExecutionStatus(
                execution_name=execution_name,
                done=False,
                succeeded=False,
                error=None,
            )

        succeeded = state == "CONDITION_SUCCEEDED"
        error_msg: str | None = None
        if not succeeded:
            error_msg = (completed or {}).get("message") or "Execution failed"

        return JobExecutionStatus(
            execution_name=execution_name,
            done=True,
            succeeded=succeeded,
            error=error_msg,
        )

    # ------------------------------------------------------------------
    # Transcript download
    # ------------------------------------------------------------------

    def download_transcript(self, session_id: str) -> dict:
        """
        Download and parse gs://bucket/{session_id}/transcript.json.
        Returns the utterances dict.
        """
        from google.cloud import storage as gcs
        client = gcs.Client(credentials=self._get_credentials())
        bucket = client.bucket(self.gcs_bucket)
        blob = bucket.blob(f"{session_id}/transcript.json")
        content = blob.download_as_text(encoding="utf-8")
        return json.loads(content)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def upload_audio_file(
        self,
        local_path: str | Path,
        session_id: str,
        progress_callback=None,
    ) -> str:
        """
        Upload a local audio file to gs://bucket/{session_id}/audio.wav.
        Optionally calls progress_callback(bytes_uploaded, total_bytes).
        Returns the GCS URI.
        """
        from google.cloud import storage as gcs
        local_path = Path(local_path)
        object_name = f"{session_id}/audio.wav"
        client = gcs.Client(credentials=self._get_credentials())
        bucket = client.bucket(self.gcs_bucket)
        blob = bucket.blob(object_name)

        total = local_path.stat().st_size
        uploaded = 0
        chunk_size = 256 * 1024  # 256 KiB

        with local_path.open("rb") as f, blob.open("wb", chunk_size=chunk_size) as out:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                uploaded += len(chunk)
                if progress_callback:
                    progress_callback(uploaded, total)

        return f"gs://{self.gcs_bucket}/{object_name}"

    def delete_session_folder(self, session_id: str) -> None:
        """Delete all objects under gs://bucket/{session_id}/."""
        from google.cloud import storage as gcs
        client = gcs.Client(credentials=self._get_credentials())
        bucket = client.bucket(self.gcs_bucket)
        blobs = list(bucket.list_blobs(prefix=f"{session_id}/"))
        if blobs:
            bucket.delete_blobs(blobs)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def session_folder_uri(self, session_id: str) -> str:
        """Returns gs://bucket/{session_id}/ URI for a session."""
        return f"gs://{self.gcs_bucket}/{session_id}/"
