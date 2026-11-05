# Project: Multimodal Vietnamese-to-H'Mong (Hmong Daw RPA) Speech-to-Speech Translation System

## Architecture
The system adopts an ultra-low latency Cascaded Speech-to-Speech (S2S) Architecture optimized for extremely low-resource language environments (< 1.5s end-to-end latency):

```
Audio Input (Mic/Stream/File)
          │
          ▼
┌───────────────────────────────────────────────┐
│ src/data/audio_preprocessor.py                │
│ - Stereo to 16kHz Mono Polyphase Resampling   │
│ - RMS Normalization (-20 dBFS, peak clipping) │
│ - Dual-stage VAD: Short-Time Energy + Silero  │
│ - 80-channel Log Mel-Spectrogram Extraction   │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│ src/models/stt_module.py                      │
│ - VietnameseSTT (Conformer-CTC Architecture)  │
│ - transcribe_chunk (Greedy Streaming Decoding)│
│ - transcribe_file (Prefix Beam Search)        │
└──────────────────────┬────────────────────────┘
                       │ Vietnamese Text Token Stream
                       ▼
┌───────────────────────────────────────────────┐
│ src/models/mt_module.py                       │
│ - Hmong Daw RPA Normalizer (8 tones preserved)│
│ - HmongTranslationModel (NLLB-200 / Seq2Seq)  │
│ - Beam Search with Repetition Penalty θ=1.2   │
└──────────────────────┬────────────────────────┘
                       │ Hmong Daw RPA Text
                       ▼
┌───────────────────────────────────────────────┐
│ src/models/tts_module.py                      │
│ - HmongVITS (Low-Resource VITS Architecture)  │
│ - RPA Phonological Text Encoder               │
│ - Stochastic / AlignNet Duration Predictor    │
│ - Normalizing Flow + HiFi-GAN Vocoder (16kHz) │
│ - synthesize(text_hmong) -> np.ndarray        │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
Audio Output (Hmong Daw 16kHz Waveform Stream / File)
```

Coordinated in real-time by `src/models/s2s_pipeline.py` (`SpeechToSpeechEngine`) under FP16 Automatic Mixed Precision (AMP) and served asynchronously via `src/api/app_stream.py` (FastAPI + RingBuffer VAD WebSocket streaming + HTML5 Web Audio UI).

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Dependency Declarations | `requirements.txt` with PyTorch 2.2+, Transformers 4.40+, soundfile, librosa, fastapi, etc. | M1 | R1 |
| 2 | Multi-stage Docker Container | `Dockerfile` for CUDA 12.1 with FFmpeg, eSpeak-ng, and multi-stage build | M1 | R1 |
| 3 | Hierarchical System Config | `configs/config.yaml` for Audio, STT, MT, TTS, and Streaming parameters | M1 | R1 |
| 4 | Audio Preprocessor & Normalization | Mono 16kHz conversion, RMS normalization (-20 dBFS) | M2 | R2 |
| 5 | Dual-Stage Voice Activity Detection | Energy-based dynamic floor + Silero VAD filtering | M2 | R2 |
| 6 | Mel-Spectrogram Extraction | Standard 80-channel Log Mel tensor extraction | M2 | R2 |
| 7 | Tri-Parallel Dataset Loader | PyTorch Dataset for (audio_vi, text_vi, text_hmong, audio_hmong) | M2 | R2 |
| 8 | Dynamic Batching & Padding | Collate function eliminating redundant zero-padding | M2 | R2 |
| 9 | Audio Augmentation | SpecAugment (time/freq masking) and Gaussian noise (SNR 10-20dB) | M2 | R2 |
| 10 | Vietnamese ASR Network | Conformer-CTC network architecture in `VietnameseSTT` | M3 | R3 |
| 11 | ASR Dual-Mode Decoding | Greedy decoding for streaming chunks, Prefix Beam Search for offline files | M3 | R3 |
| 12 | Hmong RPA Morphology Normalizer | Regex normalizer preserving 8 tones (-b, -m, -d, -j, -v, -s, -g, mid) | M3 | R3 |
| 13 | NMT Model Adaptation | Seq2Seq Transformer / NLLB-200 adaptation for Vi -> Hmong Daw RPA | M3 | R3 |
| 14 | Repetition Penalty Decoding | Beam search with repetition penalty (θ=1.2) to prevent infinite loops | M3 | R3 |
| 15 | RPA Phoneme & Text Encoder | VITS text encoder mapping RPA phonological units to prior distribution | M3 | R3 |
| 16 | Duration Predictor & MAS | Monotonic Alignment Search and duration predictor for RPA tokens | M3 | R3 |
| 17 | Flow & HiFi-GAN Vocoder | Invertible normalizing flow and multi-receptive field HiFi-GAN generator | M3 | R3 |
| 18 | VITS Waveform Synthesis | `synthesize(text_hmong) -> np.ndarray` at 16kHz | M3 | R3 |
| 19 | S2S Pipeline Engine | `SpeechToSpeechEngine` coordinating Preprocessor, STT, MT, and TTS | M3 | R3 |
| 20 | Streaming Low-Latency Generator | Generator yielding audio chunks under FP16 AMP and CUDA cache management | M3 | R3 |
| 21 | STT Training Pipeline | `scripts/train_stt.py` with CTC loss, AMP, and CER checkpointing | M4 | R4 |
| 22 | MT Training Pipeline | `scripts/train_mt.py` with Label Smoothing CE loss and SacreBLEU tracking | M4 | R4 |
| 23 | Back-Translation Synthesizer | Synthetic pseudo-parallel generation from monolingual Hmong text | M4 | R4 |
| 24 | TTS Multi-Objective Training | `scripts/train_tts.py` with Mel, KL, Duration, and GAN losses | M4 | R4 |
| 25 | Academic Evaluation Benchmark | `scripts/eval.py` assessing WER, CER, SacreBLEU, BLEU-4, chrF++, RTF, Latency | M5 | R5 |
| 26 | LaTeX & Markdown Reporting | Export formatted terminal tables and LaTeX `table_results.tex` | M5 | R5 |
| 27 | RESTful Model API Endpoints | POST /api/v1/stt, /api/v1/translate, /api/v1/tts, /api/v1/s2s in FastAPI | M6 | R6 |
| 28 | Real-Time WebSocket Streaming | `/ws/s2s_live` with RingBuffer VAD and audio chunk streaming | M6 | R6 |
| 29 | HTML5 Web Demo Interface | Zero-dependency browser interface with Web Audio API recording/playback | M6 | R6 |
| 30 | Scientific Research Monograph | `docs/BAO_CAO_NCKH.md` 5 chapters (>= 3,000 words, MOET/BKHCN standard) | M7 | R7 |
| 31 | Static Syntax Compilation Check | `python -m py_compile` across all `.py` files in `src/` and `scripts/` | M8 | R8 |
| 32 | Unit & Forward-Pass Verification | Instantiation and execution verification of all modules | M8 | R8 |
| 33 | Forensic Integrity Verification | Clean forensic audit (zero stubs, zero mocks, zero pass/TODO) | M8 | R8 |
| 34 | Production Vietnamese STT Engine | Wav2Vec2 pretrained integration in `VietnameseSTT`, accurate diacritics, safe MP3/WAV audio loading | P2-M1 | Followup-R1 |
| 35 | End-to-End S2S Pipeline & Streaming Hardening | Checkpoint auto-loading in `s2s_pipeline.py`, WebSocket single audio delivery, sample-based VAD | P2-M2 | Followup-R2 |
| 36 | Mau_Prompt Standardization & Governance | Root `RULES.md`, `ARCHITECTURE.md`, `SPEC.md`, `README.md`, and `.github/workflows/ci.yml` | P2-M3 | Followup-R3 |
| 37 | Realistic Vietnamese Speech Dataset & E2E Testing | Authentic 16kHz speech WAVs and automated pipeline tests (`test_real_speech_stt.py`, `test_e2e_audio_pipeline.py`) | P2-M4 | Followup-Verification |
| 38 | Phase 2 Review, Adversarial Stress & Forensic Audit | Independent Reviewers, Challengers, and Forensic Auditor CLEAN verdict | P2-M5 | Followup-AC |
| 39 | Multi-Domain Bilingual Conversational Lexicon | Curated 400+ entry lexicon across 9 daily domains in `data/lexicon/bilingual_conversational_dict.json` | P3-M1 | R2 |
| 40 | Deterministic Number Synthesizer | `HmongNumberSynthesizer` for numerals 0-1,000,000 in digits and text | P3-M1 | R2 |
| 41 | Greedy Longest-Match Phrase Segmenter & Rule Engine | N-gram sliding window segmenter in `HmongTranslationModel` | P3-M1 | R2 |
| 42 | Constrained Neural Seq2Seq Decoding | Transformer decoding with `<unk>` logit clamping ($P(\text{<unk>}) = 0$) | P3-M1 | R2 |
| 43 | Zero-<unk> Enforcer & 8-Tone Validator | Post-generation cleaner, compound-hyphen validator, and 8-tone preservation | P3-M1 | R2 |
| 44 | VITS DC Offset Removal & Target RMS Normalization | 2-stage DC zeroing, target RMS (-18 dBFS ~ 0.1259) and 0.95 peak ceiling in `HmongVITS.synthesize` | P3-M2 | R3 |
| 45 | Compliant RIFF/WAV Header & PCM16 Packaging | Standard 44-byte RIFF linear PCM in `synthesize_to_bytes(format="wav")` | P3-M2 | R3 |
| 46 | Web UI Vietnamese Text Input Field & Controls | `<textarea id="viTextInput">`, `#translateTextBtn`, `#clearTextBtn`, sample chips in `INDEX_HTML` | P3-M3 | R1 |
| 47 | Dedicated T2S & Unified Dual Input Endpoints | `POST /api/v1/t2s` and dual-input text handling in `POST /api/v1/s2s` | P3-M3 | R1 |
| 48 | Web Audio Player Autoplay Mitigation & Replay UI | User gesture unlock, audioPlayer currentTime=0 replay, autoplay notice, `<audio controls>` in `INDEX_HTML` | P3-M3 | R3 |
| 49 | Static Frontend Re-Export | Updated `frontend/index.html` via `scripts/export_frontend.py` | P3-M3 | R1, R3 |
| 50 | Authentic YouTube Speech Extraction Pipeline | Extract authentic Vietnamese speech segments from YouTube via `yt-dlp` android client & FFmpeg | P3-M4 | Verification |
| 51 | E2E Automated Verification Test Suites | Regression & acceptance test suites in `tests/` verifying R1 dual input, R2 0-<unk> 8-tones, R3 RMS [-20, -16] dBFS | P3-M4 | Verification |
| 52 | Phase 3 Consortium Review, Challenge & Forensic Victory Audit | Independent Reviewers (APPROVE), Challengers (PASS), and Forensic Auditor (CLEAN) | P3-M5 | AC |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Environment & Configs | `requirements.txt`, `Dockerfile`, `configs/config.yaml` | none | DONE |
| M2 | Data Pipeline & Preprocessing | `src/data/audio_preprocessor.py`, `src/data/dataset_loader.py` | M1 | DONE |
| M3 | Core AI Model Modules | `src/models/stt_module.py`, `src/models/mt_module.py`, `src/models/tts_module.py`, `src/models/s2s_pipeline.py` | M1, M2 | DONE |
| M4 | Training Pipelines | `scripts/train_stt.py`, `scripts/train_mt.py`, `scripts/train_tts.py` | M2, M3 | DONE |
| M5 | Academic Evaluation Suite | `scripts/eval.py`, `table_results.tex` | M3 | DONE |
| M6 | Streaming API Server & Demo UI | `src/api/app_stream.py` with REST, WebSocket `/ws/s2s_live`, and HTML5 UI | M3 | DONE |
| M7 | Scientific Research Monograph | `docs/BAO_CAO_NCKH.md` (5 chapters, >= 3,000 words) | M0-M6 | DONE |
| M8 | Final Integration & Victory Audit | Static compile, forward pass verification, E2E testing, forensic audit | M1-M7 | DONE |
| P2-M1 | Production Vietnamese STT Upgrade | `src/models/stt_module.py`, `src/data/audio_preprocessor.py`, `configs/config.yaml` | none | DONE |
| P2-M2 | S2S Pipeline & Streaming Integration | `src/models/s2s_pipeline.py`, `src/api/app_stream.py` | P2-M1 | DONE |
| P2-M3 | Mau_Prompt Governance & Documentation | Root `RULES.md`, `ARCHITECTURE.md`, `SPEC.md`, `README.md`, `.github/workflows/ci.yml` | none | DONE |
| P2-M4 | Realistic Speech Audio Dataset & E2E Tests | `tests/data/*.wav`, `tests/test_real_speech_stt.py`, `tests/test_e2e_audio_pipeline.py` | P2-M1, P2-M2 | DONE |
| P2-M5 | Phase 2 Review, Challenge & Forensic Audit | Verification across all acceptance criteria | P2-M1 to P2-M4 | DONE |
| P3-M1 | MT Quality, Vocabulary & 8-Tone Preservation | `src/models/mt_module.py`, `data/lexicon/bilingual_conversational_dict.json` | none | DONE |
| P3-M2 | Audible H'Mong TTS Synthesis & Normalization | `src/models/tts_module.py` (DC removal, RMS scaling -18 dBFS, ceiling 0.95) | none | DONE |
| P3-M3 | Dual Input Methods & Web Audio Playback UI | `src/api/app_stream.py`, `INDEX_HTML`, `scripts/export_frontend.py` | P3-M1, P3-M2 | DONE |
| P3-M4 | YouTube Speech Pipeline & E2E Test Suite | `tests/test_real_speech_stt.py`, `tests/test_e2e_audio_pipeline.py`, `tests/test_api_stream.py`, `tests/data/yt_speech/` | P3-M1, P3-M2, P3-M3 | DONE |
| P3-M5 | Phase 3 Review, Stress Challenge & Forensic Audit | Full Consortium Gate Verification across R1, R2, R3 and Acceptance Criteria | P3-M1 to P3-M4 | DONE |



---

## Interface Contracts

### AudioPreprocessor ↔ Models
- AudioPreprocessor output: `torch.Tensor` of shape `[channels=1, samples]` at 16,000 Hz, RMS normalized to -20 dBFS.
- Log Mel-Spectrogram: `torch.Tensor` of shape `[B, n_mels=80, T_mel]`.

### VietnameseSTT ↔ HmongTranslationModel
- `transcribe_file(audio_path: str) -> str`: Returns normalized Vietnamese text string.
- `transcribe_chunk(audio_array_buffer: np.ndarray) -> str`: Returns interim or finalized Vietnamese text segment.

### HmongTranslationModel ↔ HmongVITS
- `translate(text_vi: str) -> str`: Returns normalized Hmong Daw RPA string (8 tone markers preserved, lowercased, whitespace normalized).
- `translate_batch(texts: List[str]) -> List[str]`.

### HmongVITS ↔ S2S Engine / Streaming API
- `synthesize(text_hmong: str) -> np.ndarray`: Returns 1D float32 numpy array normalized to $[-1.0, 1.0]$ at 16,000 Hz.
- `synthesize_to_bytes(text_hmong: str) -> bytes`: Returns raw 16-bit PCM bytes or WAV container.

### SpeechToSpeechEngine
- `process_file(audio_vi_path: str) -> Tuple[str, str, np.ndarray]`: Returns `(text_vi, text_hmong, audio_hmong_wav)`.
- `process_stream(audio_generator: Iterator[bytes]) -> Iterator[bytes]`: Yields synthesized Hmong audio PCM chunks in real-time.

---

## Code Layout
```
d:/NCKH/
├── configs/
│   └── config.yaml
├── requirements.txt
├── Dockerfile
├── table_results.tex
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── audio_preprocessor.py
│   │   └── dataset_loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── stt_module.py
│   │   ├── mt_module.py
│   │   ├── tts_module.py
│   │   └── s2s_pipeline.py
│   └── api/
│       ├── __init__.py
│       └── app_stream.py
├── scripts/
│   ├── __init__.py
│   ├── train_stt.py
│   ├── train_mt.py
│   ├── train_tts.py
│   └── eval.py
├── docs/
│   └── BAO_CAO_NCKH.md
└── tests/
    ├── __init__.py
    ├── test_tier1_feature.py
    ├── test_tier2_boundary.py
    ├── test_tier3_combinations.py
    └── test_tier4_scenarios.py
```
