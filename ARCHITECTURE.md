# KIẾN TRÚC HỆ THỐNG DỰ ÁN (PROJECT ARCHITECTURE)

Tài liệu này là nguồn sự thật duy nhất (Single Source of Truth) về cấu trúc dự án và thiết kế phân tầng của Hệ thống Dịch máy Tiếng nói Đa phương thức Tiếng Việt -> Tiếng H'Mông Trắng (Hmong Daw RPA).
Mọi tác vụ sinh code, tạo file mới hoặc tái cấu trúc (refactor) của AI bắt buộc phải tuân theo cấu trúc dưới đây.

---

## 1. TỔNG QUAN HỆ THỐNG
- **Tên dự án:** Multimodal Vietnamese-to-H'Mong (Hmong Daw RPA) Speech-to-Speech Translation System
- **Mục tiêu cốt lõi:** Dịch tiếng nói hai chiều thời gian thực từ Tiếng Việt sang Tiếng H'Mông Trắng (chữ viết RPA) thích ứng môi trường thiếu ngữ liệu (Extremely Low-Resource), đạt độ trễ thời gian thực end-to-end < 1.5s và Real-Time Factor (RTF) < 0.1x.
- **Tech Stack chính:**
  - Runtime / Ngôn ngữ: Python 3.10+ / PyTorch 2.2+ (CUDA 12.1 / cuDNN 8)
  - Core Deep Learning & NLP: Torchaudio, Transformers 4.40+, Datasets, PEFT, Tokenizers
  - Xử lý Tín hiệu Âm thanh: Librosa, SoundFile, PyWorld, Phonemizer, eSpeak-ng
  - Web Server & Streaming: FastAPI 0.110+, Uvicorn, WebSockets, Pydantic v2
  - Đánh giá Học thuật: JiWER, SacreBLEU, Matplotlib, Tabulate, PyYAML

---

## 2. CÂY THƯ MỤC VÀ TRÁCH NHIỆM TỪNG PHẦN
*AI không được tự ý tạo thêm các thư mục nằm ngoài cấu trúc đã định nghĩa dưới đây:*

```text
d:/NCKH/
├── configs/
│   └── config.yaml             # Cấu hình phân cấp toàn hệ thống (Audio, STT, MT, TTS, Streaming, Training)
├── src/
│   ├── api/                    # Tầng giao tiếp mạng, định tuyến REST và WebSocket streaming
│   │   ├── __init__.py
│   │   └── app_stream.py       # FastAPI application, endpoint /ws/s2s_live, HTML5 UI demo
│   ├── data/                   # Tầng xử lý dữ liệu và tiền xử lý âm học
│   │   ├── __init__.py
│   │   ├── audio_preprocessor.py # Resampling 16kHz, RMS normalization, VAD, Mel-spectrogram
│   │   └── dataset_loader.py   # Tri-parallel PyTorch Dataset, Dynamic Batching, SpecAugment
│   └── models/                 # Tầng logic nghiệp vụ & Mô hình học sâu lõi (Core Engine)
│       ├── __init__.py
│       ├── stt_module.py       # VietnameseSTT (Conformer-CTC acoustic model & decoding)
│       ├── mt_module.py        # HmongTranslationModel (NLLB-200, RPA normalizer, repetition penalty)
│       ├── tts_module.py       # HmongVITS (RPA text encoder, duration predictor, HiFi-GAN)
│       └── s2s_pipeline.py     # SpeechToSpeechEngine (Điều phối cascaded pipeline, AMP, streaming)
├── scripts/                    # Kịch bản thực thi độc lập (Huấn luyện & Đánh giá)
│   ├── __init__.py
│   ├── train_stt.py            # Huấn luyện fine-tune CTC loss
│   ├── train_mt.py             # Huấn luyện Seq2Seq MT với label smoothing
│   ├── train_tts.py            # Huấn luyện VITS đa mục tiêu (Mel, KL, Duration, GAN)
│   ├── eval.py                 # Đánh giá học thuật toàn diện (WER, CER, BLEU, RTF, Latency, LaTeX)
│   └── export_frontend.py      # Xuất giao diện web độc lập
├── tests/                      # Bộ kiểm thử 4 phân tầng (4-Tier Verification Hierarchy)
│   ├── __init__.py
│   ├── test_utils.py           # Tiện ích test, fixtures, bộ sinh tín hiệu giả lập
│   ├── test_tier1_feature.py   # Category-Partition Unit & Feature tests (36 tests)
│   ├── test_tier2_boundary.py  # Boundary Value Analysis & Edge cases (14 tests)
│   ├── test_tier3_combinations.py # Pairwise Integration tests (6 tests)
│   ├── test_tier4_scenarios.py # Real-World Workloads & Latency benchmarks (7 tests)
│   ├── test_api_stream.py      # Kiểm thử FastAPI REST & WebSocket endpoints
│   └── test_empirical_streaming_challenge.py # Kiểm thử chịu tải streaming đa luồng
├── docs/                       # Tài liệu học thuật và báo cáo khoa học
│   └── BAO_CAO_NCKH.md         # Báo cáo NCKH toàn văn chuẩn Bộ GD&ĐT (>= 3.000 từ, 5 chương)
├── checkpoints/                # Trọng số mô hình đã huấn luyện (*.pt)
├── Dockerfile                  # Multi-stage Docker container tối ưu CUDA 12.1
├── requirements.txt            # Danh mục thư viện phụ thuộc
└── table_results.tex           # Bảng số liệu thực nghiệm xuất ra định dạng LaTeX
```

**Quy tắc phân tầng:**
1. `src/api/` chỉ nhận HTTP/WebSocket request, validate payload qua Pydantic schema và gọi sang `src/models/`. Không viết trực tiếp thuật toán deep learning hoặc DSP tại đây.
2. `src/data/` xử lý tín hiệu mộc, audio DSP thuần túy, và load dữ liệu. Không import ngược từ `src/models/` hay `src/api/`.
3. `src/models/` chứa toàn bộ mạng nơ-ron và logic dịch. Mỗi module độc lập có thể kiểm thử riêng biệt.
4. `src/models/s2s_pipeline.py` là facade điều phối duy nhất giữa các module STT, MT, và TTS.

---

## 3. LUỒNG DỮ LIỆU CHUẨN (REAL-TIME DATA FLOW)

Mọi luồng xử lý Speech-to-Speech thời gian thực đều tuân thủ các bước tuần tự:

```
[Audio Input: 16kHz PCM / Mic / File]
                 │
                 ▼
1. Validation & Preprocessing (src/data/audio_preprocessor.py)
   - Kiểm tra định dạng ➔ Downmix Mono ➔ Polyphase Resampling (16kHz)
   - RMS Normalize (-20 dBFS, peak clipping protection)
   - Dual-stage VAD (Short-Time Energy + Silero VAD) lọc khoảng lặng
                 │
                 ▼
2. Vietnamese Speech Recognition (src/models/stt_module.py)
   - Trích xuất 80-channel Log Mel-spectrogram
   - Conformer-CTC Forward Pass ➔ Greedy Decoding (streaming) / Beam Search (offline)
   - Output: Chuỗi văn bản Tiếng Việt chuẩn có dấu
                 │
                 ▼
3. Machine Translation & Normalization (src/models/mt_module.py)
   - Tiền xử lý văn bản ➔ NLLB-200 Seq2Seq Inference
   - Repetition Penalty Beam Search (θ=1.2) chống lặp từ vô hạn
   - Bảo toàn chuẩn 8 phụ âm thanh điệu RPA (-b, -m, -d, -j, -v, -s, -g, và thanh bằng)
   - Output: Chuỗi văn bản Tiếng H'Mông Trắng chuẩn RPA
                 │
                 ▼
4. Speech Synthesis (src/models/tts_module.py)
   - RPA Phoneme Text Encoder ➔ AlignNet Duration Predictor
   - Normalizing Flow ➔ HiFi-GAN Vocoder Upsampling (256x)
   - Output: Mảng sóng âm float32 16kHz trong khoảng [-1.0, 1.0]
                 │
                 ▼
5. Streaming & Response Delivery (src/api/app_stream.py)
   - Đóng gói WAV container / Raw PCM bytes
   - Stream audio chunks qua WebSocket `/ws/s2s_live` tới Client với độ trễ < 1.5s
```

---

## 4. QUY ƯỚC ĐẶT TÊN & TỔ CHỨC FILE (CONVENTIONS)
- **Tên file & module:** Dùng `snake_case` (ví dụ: `audio_preprocessor.py`, `stt_module.py`).
- **Tên Class:** Dùng `PascalCase` (ví dụ: `VietnameseSTT`, `HmongTranslationModel`, `SpeechToSpeechEngine`).
- **Tên hàm & biến:** Dùng `snake_case` (ví dụ: `transcribe_chunk`, `synthesize`, `rms_target_db`).
- **Hằng số (Constants):** Dùng `UPPER_SNAKE_CASE` (ví dụ: `SAMPLE_RATE = 16000`, `VIETNAMESE_CTC_VOCAB`).

---

## 5. CÁC HÀM TIỆN ÍCH DÙNG CHUNG CÓ SẴN (SHARED UTILS)
Khi triển khai tính năng mới hoặc kiểm thử, bắt buộc tái sử dụng:
- **Audio Preprocessor:** Khởi tạo từ `src.data.audio_preprocessor.AudioPreprocessor`
- **Config Loader:** Tải cấu hình từ `tests.test_utils.load_project_config()` hoặc đọc trực tiếp `configs/config.yaml`
- **Synthetic Signal Generator:** Sinh audio kiểm thử từ `tests.test_utils.generate_synthetic_audio()`
- **RMS Measurement:** Đo cường độ âm thanh chuẩn qua `tests.test_utils.calculate_rms_dbfs()`
