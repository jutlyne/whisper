from pathlib import Path
from uuid import uuid4

from faster_whisper import WhisperModel

from .runtime_paths import PYTHON_MODELS_DIR


_BASE_INTERVIEW_TERMS = (
    # Vai trò & cấp độ
    "Junior, Senior, Mid-level, Fresher, Intern, Trainee, Lead, Principal, Manager, "
    "Tech Lead, Fullstack, Frontend, Backend, DevOps, QA, Tester, BA, "
    # Ngôn ngữ & framework
    ".NET, C#, ASP.NET, ASP.NET Core, Entity Framework, LINQ, "
    "Java, Spring Boot, Kotlin, "
    "Python, Django, FastAPI, Flask, "
    "JavaScript, TypeScript, Node.js, Express, NestJS, "
    "ReactJS, React, Next.js, NextJS, Vue.js, Angular, "
    "PHP, Laravel, "
    "Golang, Go, Rust, "
    # Mobile
    "Flutter, Dart, Swift, SwiftUI, Android, iOS, React Native, "
    # Database
    "SQL, MySQL, PostgreSQL, SQL Server, MongoDB, Redis, Elasticsearch, Oracle, "
    "ORM, migration, stored procedure, index, query, join, "
    # Cloud & DevOps
    "Azure, AWS, GCP, Docker, Kubernetes, CI/CD, GitHub Actions, Jenkins, "
    "microservice, monolith, serverless, container, pipeline, "
    # Khái niệm kỹ thuật
    "API, REST, RESTful, GraphQL, gRPC, WebSocket, OAuth, JWT, "
    "SOLID, DDD, Clean Architecture, MVC, MVVM, Repository Pattern, "
    "unit test, integration test, code review, pull request, Git, "
    "Agile, Scrum, Kanban, sprint, "
    # Tuyển dụng
    "CV, portfolio, offer, onboard, thử việc, chính thức, "
    "part-time, full-time, freelance, remote, hybrid, "
    "lương, gross, net, thưởng, KPI"
)


class WhisperService:
    def __init__(self) -> None:
        self._model_name = "large-v3"
        self._download_root = PYTHON_MODELS_DIR / "faster-whisper"
        self._model: WhisperModel | None = None

    def set_model(self, model_name: str) -> None:
        if model_name != self._model_name:
            self._model_name = model_name
            self._model = None  # force reload

    def ensure_runtime(self) -> None:
        self._download_root.mkdir(parents=True, exist_ok=True)

    def generate_transcript(
        self,
        *,
        audio_path: str,
        language: str = "auto",
        initial_prompt: str | None = None,
    ) -> dict:
        self.ensure_runtime()
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise RuntimeError("Khong tim thay file audio de transcript.")

        combined_prompt = _BASE_INTERVIEW_TERMS
        if initial_prompt:
            combined_prompt = f"{combined_prompt}, {initial_prompt}"

        segments, info = self._get_model().transcribe(
            str(audio_file),
            language=None if language == "auto" else language,
            vad_filter=True,
            beam_size=5,
            initial_prompt=combined_prompt,
        )
        return self._build_transcript(
            segments=list(segments),
            detected_language=getattr(info, "language", "auto"),
        )

    @staticmethod
    def _build_transcript(
        *,
        segments: list[object],
        detected_language: str,
    ) -> dict:
        utterances = []
        for index, segment in enumerate(segments):
            utterances.append(
                {
                    "id": f"segment-{index + 1}",
                    "speaker": None,
                    "start": int(segment.start * 1000),
                    "end": int(segment.end * 1000),
                    "text": segment.text.strip(),
                    "timestampFrom": WhisperService._format_timestamp(
                        segment.start
                    ),
                    "timestampTo": WhisperService._format_timestamp(segment.end),
                }
            )

        return {
            "id": f"whisper-{uuid4()}",
            "text": " ".join(item["text"] for item in utterances).strip(),
            "utterances": utterances,
            "language": detected_language or "auto",
            "provider": "faster-whisper",
            "hasSpeakerDiarization": False,
        }

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                self._model_name,
                device="cpu",
                compute_type="int8",
                download_root=str(self._download_root),
            )
        return self._model

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        total_milliseconds = int(max(seconds, 0) * 1000)
        minutes, milliseconds = divmod(total_milliseconds, 60_000)
        secs, milliseconds = divmod(milliseconds, 1000)
        return f"{minutes:02d}:{secs:02d}.{milliseconds:03d}"
