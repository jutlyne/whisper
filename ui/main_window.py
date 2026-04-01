import threading
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from services.audio_capture import AudioCaptureService
from services.audio_processing import normalize_audio
from services.diarization_service import DiarizationService
from services.session_store import SessionStore
from services.whisper_service import WhisperService


class MainWindow(QMainWindow):
    _stop_success = Signal(str)
    _stop_failed = Signal(str)
    _transcript_success = Signal(str, str)  # session_id, note
    _transcript_failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Python Meeting Audio Recorder")
        self.resize(1200, 820)

        self.store = SessionStore()
        self.audio_capture = AudioCaptureService()
        self.whisper = WhisperService()
        self.diarization = DiarizationService()
        self.active_session_id: str | None = None

        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("Meeting notes")

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(
            [
                "mixed",
                "microphoneOnly",
                "systemOnly",
            ]
        )

        self.microphone_combo = QComboBox()
        self.system_combo = QComboBox()
        self.status_label = QLabel("Idle")
        self.note_label = QLabel(
            "Audio-only Python app. Start with microphone, system audio, or a mixed strategy."
        )
        self.note_label.setWordWrap(True)

        self.sessions_list = QListWidget()
        self.transcript_preview = QTextEdit()
        self.transcript_preview.setReadOnly(True)
        self.error_output = QTextEdit()
        self.error_output.setReadOnly(True)
        self.error_output.setMaximumHeight(160)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["large-v3", "large-v2", "medium", "small", "base"])
        self.model_combo.setToolTip("Model lon hon = chinh xac hon nhung cham hon va nang hon")

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText(
            "Vi du: .NET, React, JavaScript, junior, senior, CV, part-time, intern"
        )
        self.prompt_input.setToolTip("Goi y tu dien ky thuat de Whisper nhan dang chinh xac hon")

        self.n_speakers_spin = QSpinBox()
        self.n_speakers_spin.setMinimum(0)
        self.n_speakers_spin.setMaximum(10)
        self.n_speakers_spin.setValue(0)
        self.n_speakers_spin.setSpecialValueText("Auto")
        self.n_speakers_spin.setToolTip("So nguoi noi (0 = tu dong)")

        self.refresh_button = QPushButton("Refresh devices")
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.transcribe_button = QPushButton("Generate transcript")
        self.stop_button.setEnabled(False)

        self.refresh_button.clicked.connect(self.refresh_devices)
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.transcribe_button.clicked.connect(self.generate_transcript)
        self.sessions_list.currentItemChanged.connect(self.render_selected_session)
        self._stop_success.connect(self._on_stop_success)
        self._stop_failed.connect(self._on_stop_failed)
        self._transcript_success.connect(self._on_transcript_success)
        self._transcript_failed.connect(self._on_transcript_failed)

        self._build_layout()
        self.refresh_devices()
        self.load_sessions()
        self._update_diarization_note()

    def _build_layout(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        layout.addWidget(QLabel("Recording title"))
        layout.addWidget(self.title_input)
        layout.addWidget(QLabel("Capture strategy"))
        layout.addWidget(self.strategy_combo)
        layout.addWidget(QLabel("Microphone"))
        layout.addWidget(self.microphone_combo)
        layout.addWidget(QLabel("System audio / loopback"))
        layout.addWidget(self.system_combo)

        layout.addWidget(QLabel("Whisper model"))
        layout.addWidget(self.model_combo)
        layout.addWidget(QLabel("Initial prompt (tu dien ky thuat, ten rieng)"))
        layout.addWidget(self.prompt_input)

        speakers_row = QHBoxLayout()
        speakers_row.addWidget(QLabel("So nguoi noi:"))
        speakers_row.addWidget(self.n_speakers_spin)
        speakers_row.addStretch()
        layout.addLayout(speakers_row)

        buttons = QHBoxLayout()
        buttons.addWidget(self.refresh_button)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.transcribe_button)
        layout.addLayout(buttons)

        layout.addWidget(QLabel("Status"))
        layout.addWidget(self.status_label)
        layout.addWidget(QLabel("Notes"))
        layout.addWidget(self.note_label)
        layout.addWidget(QLabel("Sessions"))
        layout.addWidget(self.sessions_list)
        layout.addWidget(QLabel("Transcript preview"))
        layout.addWidget(self.transcript_preview)
        layout.addWidget(QLabel("Errors"))
        layout.addWidget(self.error_output)

        self.setCentralWidget(root)

    def _set_error(self, message: str) -> None:
        self.error_output.setPlainText(message)

    def _set_note(self, message: str) -> None:
        self.note_label.setText(message)

    def _selected_name(self, combo: QComboBox) -> str:
        data = combo.currentData()
        if not data:
            return ""
        return str(data.get("name", ""))

    def _selected_id(self, combo: QComboBox) -> str:
        data = combo.currentData()
        if not data:
            return ""
        return str(data.get("id", ""))

    def refresh_devices(self) -> None:
        self._set_error("")
        devices = self.audio_capture.list_audio_devices()
        microphones = devices.get("microphones", [])
        systems = devices.get("systems", [])

        self.microphone_combo.clear()
        self.system_combo.clear()

        if not microphones:
            self.microphone_combo.addItem("No microphone detected", None)

        if not systems:
            self.system_combo.addItem("No system audio device detected", None)

        if not microphones and not systems:
            self._set_note(
                "Python audio backend khong tim thay microphone hoac loopback device."
            )
            return

        for device in microphones:
            self.microphone_combo.addItem(device["name"], device)

        for device in systems:
            self.system_combo.addItem(device["name"], device)

        self._set_note(
            "Device list da duoc refresh tu Python-native audio backend."
        )

    def load_sessions(self) -> None:
        self.sessions_list.clear()
        for session in self.store.list_sessions():
            item = QListWidgetItem(
                f"{session['title']} [{session['status']}]"
            )
            item.setData(32, session["id"])
            self.sessions_list.addItem(item)

    def start_recording(self) -> None:
        try:
            strategy = self.strategy_combo.currentText()
            microphone_name = self._selected_name(self.microphone_combo)
            system_name = self._selected_name(self.system_combo)
            microphone_id = self._selected_id(self.microphone_combo)
            system_id = self._selected_id(self.system_combo)
            title = self.title_input.text().strip() or "Python audio session"

            session = self.store.create_session(
                title=title,
                strategy=strategy,
                microphone_name=microphone_name,
                system_name=system_name,
            )
            temp_audio_path = Path(session["tempAudioPath"])

            self.audio_capture.start_recording(
                strategy=strategy,
                microphone_id=microphone_id,
                system_id=system_id,
                output_path=temp_audio_path,
            )

            self.active_session_id = session["id"]
            self.status_label.setText("Recording")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self._set_note("Dang ghi audio-only trong app Python.")
            self.load_sessions()
        except Exception as error:
            self._set_error(str(error))

    def stop_recording(self) -> None:
        if not self.active_session_id:
            return

        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.status_label.setText("Stopping...")

        session_id = self.active_session_id
        self.active_session_id = None

        def _do_stop() -> None:
            try:
                temp_audio_path = self.audio_capture.stop_recording()
                final_audio_path = self.store.build_final_audio_path(session_id)
                normalize_audio(temp_audio_path, final_audio_path)
                temp_audio_path.unlink(missing_ok=True)
                self.store.update_session(
                    session_id,
                    status="processed",
                    finishedAt=self.store._timestamp(),
                    finalAudioPath=str(final_audio_path),
                )
                self._stop_success.emit(session_id)
            except Exception as error:
                self._stop_failed.emit(str(error))

        threading.Thread(target=_do_stop, daemon=True).start()

    def _on_stop_success(self, session_id: str) -> None:
        self.status_label.setText("Processed")
        self._set_note("Phien ghi da stop. File WAV da san sang cho transcript offline.")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.load_sessions()
        self._select_session(session_id)

    def _on_stop_failed(self, message: str) -> None:
        self._set_error(message)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Idle")

    def generate_transcript(self) -> None:
        item = self.sessions_list.currentItem()
        if not item:
            self._set_error("Hay chon mot session truoc khi transcript.")
            return

        session_id = item.data(32)
        model_name = self.model_combo.currentText()
        prompt = self.prompt_input.text().strip() or None
        n_speakers = self.n_speakers_spin.value() or None

        self.transcribe_button.setEnabled(False)
        self.status_label.setText("Processing...")
        self._set_note("Dang xu ly transcript, vui long cho...")

        def _do_transcribe() -> None:
            # --- logic goc, khong thay doi ---
            try:
                session = self.store.load_session(session_id)
                self.whisper.set_model(model_name)
                transcript = self.whisper.generate_transcript(
                    audio_path=session["finalAudioPath"],
                    language="auto",
                    initial_prompt=prompt,
                )
                diarization_result = self.diarization.run_diarization(
                    session["finalAudioPath"],
                    utterances=transcript.get("utterances"),
                    n_speakers=n_speakers,
                )
                if diarization_result.get("available") and diarization_result.get("segments"):
                    speaker_map = {
                        seg["id"]: seg["speaker"]
                        for seg in diarization_result["segments"]
                    }
                    for utt in transcript["utterances"]:
                        utt["speaker"] = speaker_map.get(utt["id"])
                    transcript["hasSpeakerDiarization"] = True

                transcript_path = self.store.save_transcript(session_id, transcript)
                self.store.update_session(
                    session_id,
                    diarizationPath=diarization_result.get("reason"),
                    transcriptPath=str(transcript_path),
                )
                note = diarization_result.get("reason", "Transcript xong.")
                self._transcript_success.emit(session_id, note)
            except Exception as error:
                self._transcript_failed.emit(str(error))
            # --- ket thuc logic goc ---

        threading.Thread(target=_do_transcribe, daemon=True).start()

    def _on_transcript_success(self, session_id: str, note: str) -> None:
        self.status_label.setText("Idle")
        self._set_note(note)
        self.transcribe_button.setEnabled(True)
        self.load_sessions()
        self._select_session(session_id)

    def _on_transcript_failed(self, message: str) -> None:
        self._set_error(message)
        self.status_label.setText("Idle")
        self.transcribe_button.setEnabled(True)

    def render_selected_session(self) -> None:
        item = self.sessions_list.currentItem()
        if not item:
            self.transcript_preview.clear()
            return

        session = self.store.load_session(item.data(32))
        transcript_path = session.get("transcriptPath")
        if not transcript_path:
            self.transcript_preview.setPlainText(
                f"Audio file: {session.get('finalAudioPath') or 'Pending'}"
            )
            return

        transcript = self.store.read_transcript(transcript_path)
        preview_lines = []
        for utterance in transcript.get("utterances", []):
            speaker = utterance.get("speaker") or "Unknown"
            timestamp = ""
            if utterance.get("timestampFrom") and utterance.get("timestampTo"):
                timestamp = f"[{utterance['timestampFrom']} - {utterance['timestampTo']}] "
            preview_lines.append(f"{timestamp}[{speaker}] {utterance['text']}")
        self.transcript_preview.setPlainText("\n".join(preview_lines))

    def _select_session(self, session_id: str) -> None:
        for index in range(self.sessions_list.count()):
            item = self.sessions_list.item(index)
            if item.data(32) == session_id:
                self.sessions_list.setCurrentItem(item)
                return

    def _update_diarization_note(self) -> None:
        probe = self.diarization.probe()
        if not probe.available:
            self._set_note(probe.reason)
