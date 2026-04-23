# Realtime Speech Translation + Offline Recorder

Repo này gồm hai thành phần độc lập:

1. **Realtime translation service** (`container_realtime/`) — dịch vụ WebSocket deploy trên Cloud Run GPU cho meeting **Việt ↔ Nhật** song phương. Nhận audio streaming, denoise, nhận dạng tiếng nói, auto-flip dịch chiều ngược lại, trả text về theo thời gian thực.
2. **Desktop recorder app** (Python/PySide6) — ứng dụng offline trên Windows, ghi âm WAV và transcribe bằng `faster-whisper` trên CPU.

---

## 1. Realtime translation service

### Kiến trúc tổng quan

```
Client (PCM 16kHz float32 mono, 100ms frames)
      │ WebSocket (binary frames + "flush" text)
      ▼
┌───────────────────────────────────────────────┐
│ Cloud Run (GPU L4, 8 vCPU, 32GB RAM)          │
│                                               │
│  FastAPI /ws endpoint                         │
│    │                                          │
│    ▼                                          │
│  StreamingTranscriber.feed()    (per session) │
│    • Silero VAD (threshold 0.35, giọng nhỏ)   │
│    • Adaptive RMS gate (loose)                │
│    • Rolling audio buffer ≤ 25s               │
│    │                                          │
│    ▼ cứ mỗi 1s → step()                       │
│                                               │
│  Denoise (DeepFilterNet3, per step)           │
│    • resample 16k → 48k → DF3 → 16k           │
│                                               │
│  ASR step (ThreadPoolExecutor, GPU)           │
│    • detect_language giới hạn {vi, ja}/utter  │
│    • UFAL whisper_streaming.OnlineASRProcessor│
│      (LocalAgreement-2, buffer trimming)      │
│      backend: faster-whisper CT2              │
│    • Repetition + hallucination + word-loop   │
│    → committed text chunks                    │
│    │                                          │
│    ├─► partial → "partial [vi] text..."       │
│    │                                          │
│    └─► finalized → translate_buffer           │
│                                               │
│  MT flush condition:                          │
│    dấu câu . ! ? 。 ! ?    hoặc               │
│    pause ≥ 2s               hoặc              │
│    buffer ≥ 240 ký tự                         │
│    │                                          │
│    ▼ NLLB worker (CT2 int8_float16)           │
│    • auto-flip target (vi↔ja)                 │
│    → "final [vi→ja] translated"               │
│                                               │
│  GCSFuse mount /models/ ◄─────────── GCS      │
│    • whisper-large-v3-ct2/ (ưu tiên)          │
│    • whisper-large-v3-turbo-ct2/ (fallback)   │
│    • nllb-200-distilled-600M-ct2/             │
│    • silero-vad/silero_vad.jit                │
└───────────────────────────────────────────────┘
```

### WebSocket protocol

| Hướng | Loại | Nội dung |
|---|---|---|
| Client → Server | binary | Float32 PCM mono, 16kHz, 100ms (1600 samples / frame) |
| Client → Server | text | `"flush"` — báo kết thúc recording, server xử lý buffer cuối |
| Server → Client | text | `"ready"` — model đã load xong |
| Server → Client | text | `"partial [<src>] <text>"` — text đang gõ, có thể thay đổi |
| Server → Client | text | `"final [S<n> <src>→<tgt>] <translated>"` — đã commit; `S1`/`S2`/... là speaker ID auto-assigned |
| Server → Client | text | `"final [S<n> <src>] <text>"` — khi src == tgt, không dịch |
| Server → Client | text | `"done"` — đã xử lý xong flush |
| Server → Client | text | `"error: <msg>"` — lỗi runtime |

**Cách client xử lý `partial` vs `final`:**
- Khi nhận `partial`, overwrite dòng hiện tại đang hiển thị.
- Khi nhận `final`, append dòng đó vào transcript và reset buffer partial.

### Query params

| Param | Default | Ý nghĩa |
|---|---|---|
| `src_langs` | `vi,ja` | Danh sách ngôn ngữ nguồn cho phép (phân tách `,`). Detection chỉ argmax trong tập này → không nhầm sang en/zh/ko/th. Đặt `auto` để detect full 99 ngôn ngữ. |
| `output_lang` | `auto` | `auto` + đúng 2 `src_langs` → flip source ↔ target (meeting mode). Hoặc cố định: `vi`, `ja`, `en`, `zh`, `ko`. |
| `vocab` | `""` | Từ vựng chuyên ngành/tên riêng phân tách bằng `,`. Được prepend vào `initial_prompt` của Whisper → bias decoder về các token đó. Ví dụ: `vocab=API,framework,Kubernetes,React`. |
| `partials` | `0` | `1` → server gửi cả `partial` (text đang gõ) + `final`. `0` (mặc định) → chỉ gửi `final` khi người nói dừng. Bật `partials=1` nếu muốn hiển thị streaming như Google Meet. |

**Ví dụ kết nối:**

```
# Meeting VI ↔ JA (mặc định) — không cần param
wss://<host>/ws

# Meeting tech với vocabulary hint
wss://<host>/ws?vocab=API,framework,backend,frontend,React,Kubernetes

# Fix một chiều: mọi nguồn dịch sang tiếng Việt
wss://<host>/ws?output_lang=vi&src_langs=auto

# Cuộc họp 3 bên VI-JA-EN
wss://<host>/ws?src_langs=vi,ja,en&output_lang=vi
```

### Cấu trúc thư mục

```
container_realtime/
├── server.py                         # FastAPI app, lifespan model loader, /ws endpoint, 2-queue pipeline
├── vad_transcriber.py                # VadTranscriber (per-session): Silero VAD + faster-whisper + CT2 NLLB
├── Dockerfile                        # Python 3.11 + CUDA 12.4 torch + faster-whisper + ctranslate2
├── requirements.txt                  # Runtime deps
├── upload_whisper_ct2_to_gcs.yaml    # Cloud Build: faster-whisper-large-v3-turbo-ct2 → GCS
├── upload_nllb_ct2_to_gcs.yaml       # Cloud Build: NLLB-200-distilled-600M → CT2 int8 + tokenizer → GCS
└── upload_silero_vad_to_gcs.yaml     # Cloud Build: Silero VAD JIT → GCS
```

### Stack kỹ thuật

- **Speaker diarization:** [resemblyzer](https://github.com/resemble-ai/Resemblyzer) — speaker embedding + centroid matching (threshold cosine > 0.75). Tự gán ID `S1`, `S2`, ... khi gặp người mới.
- **ASR streaming policy:** [UFAL whisper_streaming](https://github.com/ufal/whisper_streaming) (LocalAgreement-2). File `whisper_online.py` fetched tại Docker build time.
- **ASR backbone:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) — `large-v3` hoặc `large-v3-turbo` float16
- **MT:** [CTranslate2](https://github.com/OpenNMT/CTranslate2) chạy NLLB-200-distilled-600M int8_float16
- **Denoise:** [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) (torch, auto-download model ~20MB lần đầu cold start)
- **VAD:** [Silero VAD](https://github.com/snakers4/silero-vad) TorchScript JIT (chạy CPU)
- **Serving:** FastAPI + uvicorn + WebSocket
- **Deploy:** Cloud Run GPU (NVIDIA L4)

### Biến môi trường

| Env | Default | Ý nghĩa |
|---|---|---|
| `ENABLE_DENOISE` | `1` | Bật DeepFilterNet. Set `0` để bỏ qua denoise (tiết kiệm ~20-40ms/segment). |
| `ENABLE_DIARIZATION` | `1` | Bật speaker diarization (resemblyzer). Set `0` → mọi câu không có speaker tag (`[S? ...]`). |
| `WHISPER_MODEL` | `auto` | `v3` → dùng `whisper-large-v3-ct2`, `turbo` → `whisper-large-v3-turbo-ct2`, `auto` → ưu tiên v3 nếu có, fallback turbo. |
| `ALLOW_CPU` | unset | Cho phép chạy không có GPU (chỉ dùng dev/test). |

---

## 2. Setup GCP (chạy một lần cho project)

Project: `whisper-494002`. Region: `asia-southeast1`.

### IAM cho user

```bash
gcloud projects add-iam-policy-binding whisper-494002 \
  --member="user:kyvc.nta@gmail.com" \
  --role="roles/cloudbuild.builds.editor"

gcloud projects add-iam-policy-binding whisper-494002 \
  --member="user:kyvc.nta@gmail.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding whisper-494002 \
  --member="user:kyvc.nta@gmail.com" \
  --role="roles/iam.serviceAccountUser"
```

### IAM cho Cloud Build service account

```bash
gcloud projects add-iam-policy-binding whisper-494002 \
  --member="serviceAccount:87603132839-compute@developer.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding whisper-494002 \
  --member="serviceAccount:87603132839-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding whisper-494002 \
  --member="serviceAccount:87603132839-compute@developer.gserviceaccount.com" \
  --role="roles/logging.logWriter"
```

### Enable APIs

```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  storage.googleapis.com \
  --project=whisper-494002
```

---

## 3. Deploy realtime service

### Bước 1 — Tạo bucket GCS chứa models

```bash
gcloud storage buckets create gs://whisper-494002-models \
  --project=whisper-494002 \
  --location=asia-southeast1
```

### Bước 2 — Upload models lên GCS (chạy song song được)

```bash
# Whisper large-v3 (accurate — RECOMMENDED cho meeting chất lượng)
gcloud builds submit --no-source \
  --config=container_realtime/upload_whisper_v3_to_gcs.yaml \
  --project=whisper-494002

# (Tùy chọn) Whisper large-v3-turbo (nhanh hơn nhưng kém accuracy ~8%)
gcloud builds submit --no-source \
  --config=container_realtime/upload_whisper_ct2_to_gcs.yaml \
  --project=whisper-494002

# NLLB-200-distilled-600M → CT2 int8 (convert trong Cloud Build)
gcloud builds submit --no-source \
  --config=container_realtime/upload_nllb_ct2_to_gcs.yaml \
  --project=whisper-494002

# Silero VAD JIT
gcloud builds submit --no-source \
  --config=container_realtime/upload_silero_vad_to_gcs.yaml \
  --project=whisper-494002
```

Kết quả kỳ vọng trên GCS:
```
gs://whisper-494002-models/whisper-large-v3-ct2/             ← ưu tiên
gs://whisper-494002-models/whisper-large-v3-turbo-ct2/       ← fallback
gs://whisper-494002-models/nllb-200-distilled-600M-ct2/
gs://whisper-494002-models/silero-vad/silero_vad.jit
```

### Bước 3 — Tạo Artifact Registry cho Docker image

```bash
gcloud artifacts repositories create whisper-realtime \
  --repository-format=docker \
  --location=us \
  --project=whisper-494002
```

### Bước 4 — Build & push Docker image

```bash
gcloud builds submit \
  --tag us-docker.pkg.dev/whisper-494002/whisper-realtime/whisper-realtime \
  --project=whisper-494002 \
  ./container_realtime
```

### Bước 5 — Deploy lên Cloud Run

```bash
gcloud run deploy whisper-realtime \
  --image us-docker.pkg.dev/whisper-494002/whisper-realtime/whisper-realtime \
  --gpu 1 --gpu-type nvidia-l4 \
  --cpu 8 --memory 32Gi \
  --no-cpu-throttling \
  --min-instances 1 \
  --execution-environment=gen2 \
  --region asia-southeast1 \
  --project=whisper-494002 \
  --allow-unauthenticated \
  --cpu-boost \
  --timeout=3600 \
  --add-volume=name=models,type=cloud-storage,bucket=whisper-494002-models \
  --add-volume-mount=volume=models,mount-path=/models \
  --set-env-vars ENABLE_DENOISE=1
```

**Tắt denoise sau khi deploy** (không cần rebuild):
```bash
gcloud run services update whisper-realtime \
  --region asia-southeast1 --project=whisper-494002 \
  --set-env-vars ENABLE_DENOISE=0
```

---

## 4. Verify & test

### Health check

```bash
curl https://<cloud-run-url>/health
```

Kỳ vọng (sau 30–90s cold start):
```json
{
  "status": "ok",
  "whisper_ready": true,
  "nllb_ct2_ready": true,
  "silero_ready": true,
  "denoise_enabled": true,
  "denoise_ready": true,
  "device": "cuda",
  "ready": true,
  "error": null
}
```

### Smoke test WebSocket

Dùng client bất kỳ gửi frame PCM float32 mono 16kHz, chunk 100ms:

```python
import asyncio, websockets, soundfile as sf
async def main():
    audio, sr = sf.read("sample.wav", dtype="float32")   # sr phải = 16000
    async with websockets.connect("wss://<host>/ws") as ws:
        print(await ws.recv())    # "ready"
        for i in range(0, len(audio), 1600):
            await ws.send(audio[i:i+1600].tobytes())
            await asyncio.sleep(0.1)
        await ws.send("flush")
        while True:
            msg = await ws.recv()
            # "partial [vi] ...", "final [vi→ja] ...", "done"
            if msg.startswith("partial"):
                print("\r" + msg, end="", flush=True)
            elif msg.startswith("final"):
                print("\n" + msg)
            elif msg == "done":
                break
asyncio.run(main())
```

---

## 5. Troubleshooting

| Triệu chứng | Nguyên nhân | Xử lý |
|---|---|---|
| `"error: model not loaded yet"` | Cold start chưa xong | Đợi 30–90s, check `/health` |
| `health.silero_ready = false` | Yaml silero upload chưa chạy hoặc mount path sai | Re-run `upload_silero_vad_to_gcs.yaml`, verify `gs://…/silero-vad/silero_vad.jit` tồn tại |
| Log `audio_queue full — dropped N segments` | Client gửi audio nhanh hơn Whisper xử lý | Kiểm tra GPU load; tăng `maxsize` trong `server.py` nếu cần |
| `CUDA out of memory` | Cả Whisper + NLLB cùng load fp16 vượt VRAM L4 (24GB) | Đổi `compute_type="int8_float16"` cho Whisper trong `server.py` |
| Dịch ngắt sau 2–3 giây | (Đã fix) Threshold RMS leo quá cao, Silero state rò rỉ | Logic hiện đã update RMS mọi chunk + reset Silero per-session |
| `flush timeout — worker may be stuck` | Worker treo khi xử lý segment cuối | Log GPU, kiểm tra Whisper model có corrupt không; restart instance |
| Partial text flicker (đổi qua đổi lại) | Bình thường với streaming — LocalAgreement-2 cần 2 runs mới commit | Flicker window < 1s là OK. Nếu > 2s: tăng `_STEP_SECONDS` trong [vad_transcriber.py](container_realtime/vad_transcriber.py) |
| Latency final cao (>2s sau khi nói xong) | Chưa gặp dấu câu kết thúc → buffer đợi pause 2s | Bình thường. Hoặc giảm `_FLUSH_PAUSE_SECONDS` nếu muốn gửi sớm hơn |
| `/health` báo `whisper_model: ""` | Không tìm thấy model folder trong `/models/` | Verify `gs://whisper-494002-models/whisper-large-v3-ct2/` tồn tại; mount path đúng |
| Output detect nhầm sang EN/ZH/KO | (Đã fix) `src_langs` param giới hạn detection về {vi, ja} | Đảm bảo client không override `src_langs=auto` |
| Whisper bịa ra "Cảm ơn các bạn đã theo dõi" / "ご視聴ありがとう" | Hallucination trên khoảng lặng / nhạc nền | Đã có blacklist trong [vad_transcriber.py](container_realtime/vad_transcriber.py) — thêm pattern mới vào `_HALLUCINATION_PATTERNS` nếu gặp câu khác |
| Denoise làm giọng bị méo / robot-like | DeepFilterNet suppress quá mạnh môi trường rất ít noise | Tắt qua env `ENABLE_DENOISE=0` |
| `denoise_ready: false` trên `/health` | DeepFilterNet download model fail ở cold start (network) | Retry instance, hoặc tắt denoise nếu không cần |

---

## 6. Desktop recorder app (Python offline)

> ⚠️ Tách biệt với realtime service ở trên. App này chạy hoàn toàn offline trên Windows, không liên quan tới Cloud Run.

PySide6 desktop app:
- `soundcard` + WASAPI loopback — capture system audio
- `soundfile` — ghi WAV
- `faster-whisper` — transcribe offline trên CPU

### Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip -r requirements.txt
```

### Run

```powershell
.\.venv\Scripts\python.exe app.py
```

### Smoke test

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

### Whisper test

```powershell
.\.venv\Scripts\python.exe scripts\test_whisper.py
# hoặc:
.\.venv\Scripts\python.exe scripts\test_whisper.py path\to\audio.wav
```

### Diarization note

`pyannote.audio` không có official Windows support, nên `services/diarization_service.py` hiện là spike có guard và fallback an toàn, chưa phải production feature.
