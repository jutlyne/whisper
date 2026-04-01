import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .runtime_paths import (
    PYTHON_METADATA_DIR,
    PYTHON_RECORDINGS_DIR,
    PYTHON_STORAGE_DIR,
    PYTHON_TEMP_DIR,
    PYTHON_TRANSCRIPTS_DIR,
)


class SessionStore:
    def __init__(self) -> None:
        self.ensure_storage()

    def ensure_storage(self) -> None:
        for target in (
            PYTHON_STORAGE_DIR,
            PYTHON_RECORDINGS_DIR,
            PYTHON_METADATA_DIR,
            PYTHON_TRANSCRIPTS_DIR,
            PYTHON_TEMP_DIR,
        ):
            target.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        *,
        title: str,
        strategy: str,
        microphone_name: str,
        system_name: str,
    ) -> dict[str, Any]:
        session_id = str(uuid4())
        session = {
            "id": session_id,
            "title": title,
            "strategy": strategy,
            "microphoneName": microphone_name,
            "systemName": system_name,
            "status": "recording",
            "createdAt": self._timestamp(),
            "updatedAt": self._timestamp(),
            "startedAt": self._timestamp(),
            "finishedAt": None,
            "tempAudioPath": str(self.build_temp_audio_path(session_id)),
            "finalAudioPath": None,
            "transcriptPath": None,
            "diarizationPath": None,
            "errors": [],
        }
        self.write_session(session)
        return session

    def list_sessions(self) -> list[dict[str, Any]]:
        self.ensure_storage()
        sessions: list[dict[str, Any]] = []
        for metadata_file in PYTHON_METADATA_DIR.glob("*.json"):
            sessions.append(self.read_json(metadata_file))
        sessions.sort(key=lambda item: item.get("updatedAt", ""), reverse=True)
        return sessions

    def load_session(self, session_id: str) -> dict[str, Any]:
        return self.read_json(self.metadata_path(session_id))

    def update_session(self, session_id: str, **updates: Any) -> dict[str, Any]:
        session = self.load_session(session_id)
        session.update(updates)
        session["updatedAt"] = self._timestamp()
        self.write_session(session)
        return session

    def append_error(
        self,
        session_id: str,
        *,
        message: str,
        status: str,
    ) -> dict[str, Any]:
        session = self.load_session(session_id)
        errors = session.get("errors", [])
        errors.append(
            {
                "message": message,
                "recordedAt": self._timestamp(),
                "status": status,
            }
        )
        session["errors"] = errors
        session["status"] = status
        session["updatedAt"] = self._timestamp()
        self.write_session(session)
        return session

    def save_transcript(
        self,
        session_id: str,
        transcript: dict[str, Any],
    ) -> Path:
        transcript_path = PYTHON_TRANSCRIPTS_DIR / f"{session_id}.json"
        transcript_path.write_text(
            json.dumps(transcript, indent=2),
            encoding="utf8",
        )
        self.update_session(
            session_id,
            status="transcribed",
            transcriptPath=str(transcript_path),
        )
        return transcript_path

    def read_transcript(self, transcript_path: str) -> dict[str, Any]:
        return self.read_json(Path(transcript_path))

    def metadata_path(self, session_id: str) -> Path:
        return PYTHON_METADATA_DIR / f"{session_id}.json"

    def build_temp_audio_path(self, session_id: str) -> Path:
        return PYTHON_TEMP_DIR / f"{session_id}.wav"

    def build_final_audio_path(self, session_id: str) -> Path:
        return PYTHON_RECORDINGS_DIR / f"{session_id}.wav"

    def write_session(self, session: dict[str, Any]) -> None:
        self.metadata_path(session["id"]).write_text(
            json.dumps(session, indent=2),
            encoding="utf8",
        )

    @staticmethod
    def read_json(target_path: Path) -> dict[str, Any]:
        return json.loads(target_path.read_text(encoding="utf8"))

    @staticmethod
    def _timestamp() -> str:
        return datetime.utcnow().isoformat() + "Z"
