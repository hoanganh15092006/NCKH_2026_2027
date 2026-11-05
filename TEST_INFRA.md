# Comprehensive End-to-End Testing Infrastructure & Architecture Specification (TEST_INFRA)

**Project**: Multimodal Vietnamese-to-H'Mong (Hmong Daw RPA) Speech-to-Speech Translation System  
**Document Track**: E2E Quality Assurance & Academic Peer Review Infrastructure  
**Author**: Academic Peer Reviewer & QA (Test Writer E2E)  
**Standard**: 4-Tier Verification Methodology (Category-Partition, Boundary Value Analysis, Pairwise/Integration, Workload/Scenario Testing)  
**Target Latency**: $\le 1,500\text{ ms}$ End-to-End  
**Target Environment**: Dual-Mode (Host Python 3.10+ / Linux CUDA 12.1 Container)  

---

## 1. Executive Summary & Testing Philosophy

The Speech-to-Speech Translation (S2ST) system bridging Vietnamese (*tiếng Việt*) and White Hmong (*Hmoob Dawb* Romanized Popular Alphabet - RPA) represents an extremely low-resource, multi-stage cascading pipeline. Because errors in early pipeline stages (e.g., misrecognized Vietnamese tones, acoustic noise) compound non-linearly through machine translation and speech synthesis, testing must not rely on trivial happy-path verifications or facade assertions.

This infrastructure enforces **strict behavioral and algorithmic testing**:
1. **Zero-Mock & Zero-Facade Integrity**: Every test exercises real mathematical operations, tensor transformations, phonetic/morphological rules, and timing measurements.
2. **Deterministic Output Derivation**: Expected outputs for acoustic features, audio normalization, regex-based RPA diacritics, and metric formulations are derived directly from acoustic physics, signal processing theory, and published linguistic standards (Bertrais et al. 1953; Popović 2017).
3. **Graceful Progressive Testability**: The test suite operates across bare developer workstations and fully provisioned GPU container environments, dynamically executing deep neural passes when PyTorch/Transformers dependencies and weights are present, while maintaining full behavioral verification of all signal processing, state machines, regex normalizers, and config parsers.

---

## 2. 4-Tier Verification Methodology

The test architecture is structured into four distinct, non-overlapping testing tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Tier 4: Real-World Scenarios, Extreme Workload & Benchmarks            │
│ (tests/test_tier4_scenarios.py: Emergency phrases, SNR 10-20dB, <1.5s) │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 3: Pairwise Chaining & Cross-Feature Integration                  │
│ (tests/test_tier3_combinations.py: Audio->STT->MT, MT->TTS, VAD->S2S)  │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 2: Boundary Value Analysis (BVA) & Edge Hardening                 │
│ (tests/test_tier2_boundary.py: Silence, clipping, >256 tokens, empty)  │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 1: Category-Partition Feature & Unit Testing                     │
│ (tests/test_tier1_feature.py: >= 5 tests per core feature)             │
└────────────────────────────────────────────────────────────────────────┘
```

### Tier 1: Category-Partition Method (Functional & Unit Level)
- **Objective**: Verify that individual atomic functions and model classes adhere to their interface contracts.
- **Partitioning Strategy**: Input domains are partitioned into equivalence classes (valid audio waveforms, valid text strings, standard sample rates, valid config parameters).
- **Coverage Mandate**: At least 5 dedicated test cases per core module:
  - Audio Preprocessing (Resampling, RMS normalization, Mel-spectrogram, VAD energy thresholding, channel conversion).
  - Vietnamese STT (Forward pass, logit output shapes, greedy decoding, beam search prefix, blank token collapse).
  - Hmong MT (RPA tone consonant preservation, case normalization, repetition penalty scoring, sequence length constraints, dictionary mapping).
  - Hmong TTS (Text phoneme encoding, duration predictor output shapes, vocoder generation, waveform normalization $[-1.0, 1.0]$, sample rate conformity).
  - Streaming VAD (RingBuffer state transitions: IDLE $\to$ SPEECH $\to$ ENDPOINT, energy dB calculation, silence chunk accumulation).
  - Hierarchical Config (YAML schema validation, type integrity, default fallback, path resolution).

### Tier 2: Boundary Value Analysis (BVA & Adversarial Robustness)
- **Objective**: Stress system boundaries and edge conditions where deep learning pipelines typically fault or produce catastrophic degradation.
- **Boundary Conditions Tested**:
  1. *Acoustic Extrema*: Absolute digital silence (zero amplitude), extreme acoustic clipping ($+6\text{ dBFS}$ to $+12\text{ dBFS}$ saturation), non-standard sampling rates ($8\text{ kHz}, 22.05\text{ kHz}, 44.1\text{ kHz}, 48\text{ kHz}$).
  2. *Textual Extrema*: Zero-length empty strings (`""`), whitespace-only strings (`"   \t\n"`), single characters (`"a"`, `"b"`), extreme sequence lengths ($> 256$ tokens exceeding transformer positional embeddings).
  3. *Morphological Edges*: Hmong RPA words with zero tone marker (mid-level tone), invalid tone markers, concatenated words without spaces, and punctuation collision.
  4. *Tensor & Numerical Edges*: Undersized audio frames ($< 300\text{ ms}$), batch size 1 vs dynamic batching boundaries, NaN and Inf propagation checks.

### Tier 3: Pairwise & Cross-Feature Integration Testing
- **Objective**: Validate the interface contracts and data compatibility between consecutive modules in the cascaded pipeline.
- **Interface Chains Tested**:
  - `AudioPreprocessor` $\longrightarrow$ `VietnameseSTT`: Checks that 16kHz mono, RMS normalized tensor conforms to STT acoustic feature requirements.
  - `VietnameseSTT` $\longrightarrow$ `HmongTranslationModel`: Verifies that decoded text strings feed into tokenization and RPA translation without loss of diacritics.
  - `HmongTranslationModel` $\longrightarrow$ `HmongVITS`: Verifies that translated Hmong RPA text (with tone consonants `-b, -m, -d, -j, -v, -s, -g`) maps directly into the VITS phoneme/character inventory.
  - `RingBufferVAD` $\longrightarrow$ `SpeechToSpeechEngine`: Validates real-time 300ms PCM chunk ingestion, silence endpointing, and streaming audio synthesis dispatch.
  - Full Cascaded Loop: End-to-end execution preserving state isolation across consecutive utterances.

### Tier 4: Workload, Scenario & Academic Benchmark Testing
- **Objective**: Replicate real-world operational environments in the Northern mountainous provinces of Vietnam (Hà Giang, Lào Cai, Lai Châu, Điện Biên).
- **Scenarios Evaluated**:
  1. *Emergency Medical & Natural Disaster Phrases*: High-stakes clinical triage and disaster alerts (e.g., "Cứu tôi với tôi bị lạc trên núi", "Tôi bị đau ngực và khó thở", "Có sạt lở đất lớn ở bản bên cạnh").
  2. *Acoustic Noise Stress*: Additive Gaussian white noise and ambient environmental noise at calibrated SNR ratios ($10\text{ dB}, 15\text{ dB}, 20\text{ dB}$).
  3. *Conversational Latency Benchmark*: Rigorous wall-clock latency measurement of every pipeline phase, asserting that End-to-End Latency $T_{\text{E2E}} \le 1,500\text{ ms}$ (P50, P90, Mean).
  4. *Continuous Multi-Turn Stream*: Sustained multi-segment ingestion validating zero memory leakage in circular buffers and GPU cache cleanup.

---

## 3. Explicit Mapping of All 33 Features from PROJECT.md

Every feature specified in `PROJECT.md` is assigned to its authoritative testing tier and verification method:

| # | Feature from PROJECT.md | Implementation Target | Primary Testing Tier | Verification Method & Test Case |
|---|-------------------------|-----------------------|----------------------|---------------------------------|
| 1 | Dependency Declarations | `requirements.txt` | Tier 1 | Parsing requirements, version specifier validity, dependency conflict check |
| 2 | Multi-stage Docker Container | `Dockerfile` | Tier 1 | Syntax parsing of multi-stage stages (builder vs runtime), non-root user check |
| 3 | Hierarchical System Config | `configs/config.yaml` | Tier 1 | Structural schema validation across system, audio, stt, mt, tts, streaming sections |
| 4 | Audio Preprocessor & Normalization | `src/data/audio_preprocessor.py` | Tier 1 & Tier 2 | RMS normalization target $\pm 0.5\text{ dBFS}$, mono downmixing, 16kHz resampling |
| 5 | Dual-Stage Voice Activity Detection | `src/data/audio_preprocessor.py` | Tier 1 & Tier 2 | Energy-based floor + VAD thresholding rejecting silence and detecting speech |
| 6 | Mel-Spectrogram Extraction | `src/data/audio_preprocessor.py` | Tier 1 | 80-channel log Mel tensor extraction, FFT windowing, time-frequency dimensions |
| 7 | Tri-Parallel Dataset Loader | `src/data/dataset_loader.py` | Tier 1 & Tier 3 | Tri-parallel tuple loading (`audio_vi`, `text_vi`, `text_hmong`, `audio_hmong`) |
| 8 | Dynamic Batching & Padding | `src/data/dataset_loader.py` | Tier 1 & Tier 2 | Collate function grouping, non-redundant zero padding, length masking |
| 9 | Audio Augmentation | `src/data/dataset_loader.py` | Tier 1 & Tier 4 | SpecAugment time/freq masking bounds and calibrated Gaussian noise (SNR 10-20dB) |
| 10 | Vietnamese ASR Network | `src/models/stt_module.py` | Tier 1 | Conformer-CTC layer instantiation, parameter initialization, forward tensor flow |
| 11 | ASR Dual-Mode Decoding | `src/models/stt_module.py` | Tier 1 & Tier 3 | Greedy chunk decoding for streaming vs Prefix Beam Search for offline files |
| 12 | Hmong RPA Morphology Normalizer | `src/models/mt_module.py` | Tier 1 & Tier 2 | Preservation of 8 tones (`-b, -m, -d, -j, -v, -s, -g`, zero), lowercasing, whitespace |
| 13 | NMT Model Adaptation | `src/models/mt_module.py` | Tier 1 & Tier 3 | Seq2Seq Transformer / NLLB-200 `vie_Latn` $\to$ `mww_Latn` translation execution |
| 14 | Repetition Penalty Decoding | `src/models/mt_module.py` | Tier 1 & Tier 2 | Beam search with repetition penalty $\theta = 1.25$ suppressing repeating n-grams |
| 15 | RPA Phoneme & Text Encoder | `src/models/tts_module.py` | Tier 1 | Mapping RPA character/phoneme sequences into VITS text encoder prior space |
| 16 | Duration Predictor & MAS | `src/models/tts_module.py` | Tier 1 | Alignment search and phoneme duration tensor prediction |
| 17 | Flow & HiFi-GAN Vocoder | `src/models/tts_module.py` | Tier 1 | Invertible normalizing flow and multi-receptive field 16kHz waveform upsampling |
| 18 | VITS Waveform Synthesis | `src/models/tts_module.py` | Tier 1 & Tier 3 | `synthesize(text_hmong) -> np.ndarray` 1D float32 output normalized to $[-1.0, 1.0]$ |
| 19 | S2S Pipeline Engine | `src/models/s2s_pipeline.py` | Tier 3 & Tier 4 | `SpeechToSpeechEngine` end-to-end coordination: Audio In $\to$ STT $\to$ MT $\to$ TTS $\to$ Audio Out |
| 20 | Streaming Low-Latency Generator | `src/models/s2s_pipeline.py` | Tier 3 & Tier 4 | Asynchronous generator yielding audio chunks with FP16 AMP and cache evacuation |
| 21 | STT Training Pipeline | `scripts/train_stt.py` | Tier 1 | CTC loss calculation, mixed precision AMP scaling, CER validation checkpointing |
| 22 | MT Training Pipeline | `scripts/train_mt.py` | Tier 1 | Label Smoothing Cross-Entropy loss $\mathcal{L}_{\text{LS-CE}}$, SacreBLEU evaluation |
| 23 | Back-Translation Synthesizer | `scripts/train_mt.py` | Tier 1 | Pseudo-parallel corpus generation from monolingual Hmong text |
| 24 | TTS Multi-Objective Training | `scripts/train_tts.py` | Tier 1 | Composite VITS loss ($\mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{kl}} + \mathcal{L}_{\text{dur}} + \mathcal{L}_{\text{adv}}$) |
| 25 | Academic Evaluation Benchmark | `scripts/eval.py` | Tier 4 | Automated scoring of WER, CER, SacreBLEU, BLEU-4, chrF++, RTF, and Latency |
| 26 | LaTeX & Markdown Reporting | `scripts/eval.py` | Tier 1 & Tier 4 | Generation of terminal markdown tables and LaTeX `table_results.tex` `booktabs` code |
| 27 | RESTful Model API Endpoints | `src/api/app_stream.py` | Tier 1 & Tier 3 | HTTP endpoints `POST /stt`, `/translate`, `/tts`, `/s2s`, `GET /health` |
| 28 | Real-Time WebSocket Streaming | `src/api/app_stream.py` | Tier 1, 3, 4 | `/ws/s2s_live` WebSocket PCM ingestion, RingBufferVAD endpointing, chunk emission |
| 29 | HTML5 Web Demo Interface | `src/api/app_stream.py` | Tier 1 | Static GET `/` HTML5 web client serving with Web Audio API recording/playback |
| 30 | Scientific Research Monograph | `docs/BAO_CAO_NCKH.md` | Tier 1 | Academic structure verification: $\ge 3,000$ words, 5 chapters, LaTeX formulas, MOS |
| 31 | Static Syntax Compilation Check | Build & CI | Tier 1 | `python -m py_compile` across all Python source modules with zero syntax errors |
| 32 | Unit & Forward-Pass Verification | Pipeline Execution | Tier 1 & Tier 3 | Instantiation of all concrete classes and execution of forward passes |
| 33 | Forensic Integrity Verification | Audit Inspection | QA Audit | Clean code audit: zero dummy mocks, zero `pass`, zero `# TODO`, zero cheating |

---

## 4. Expected Output Derivation & Mathematical Oracles

To guarantee test integrity, all expected values are derived from mathematical models, acoustic theory, or authoritative linguistic specifications:

### 4.1 Audio RMS Normalization Oracle
Given discrete signal $x \in \mathbb{R}^N$, the root-mean-square amplitude is:
$$\text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}$$
Level in decibels relative to full scale (dBFS):
$$L_{\text{dBFS}} = 20 \log_{10}(\text{RMS}(x) + \epsilon)$$
Target level is $L_{\text{target}} = -20.0\text{ dBFS}$. The required scaling gain $g$ is:
$$g = 10^{\frac{L_{\text{target}} - L_{\text{dBFS}}}{20}} = \frac{10^{-1.0}}{\text{RMS}(x)}$$
Expected Output Oracle: Any normalized waveform must satisfy $|20 \log_{10}(\text{RMS}(x_{\text{norm}})) - (-20.0)| \le 0.5\text{ dBFS}$.

### 4.2 Hmong RPA Morphological Tone Normalization Oracle
Hmong Daw syllables follow the canonical structure $C(C)V(V)T$, where the terminal consonant $T$ represents one of 8 lexical tones:
$$\mathcal{T} = \{\text{'-b'} (55), \text{'-j'} (52), \text{'-v'} (34), \text{'-s'} (21), \text{'-g'} (42̤), \text{'-m'} (21ʔ), \text{'-d'} (24), \emptyset (33)\}$$
A regex normalizer must preserve all valid tone consonants while stripping non-RPA characters, extraneous punctuation, and extra whitespace.
Expected Output Oracle:
- Input `"Kuv mus pem roob b"` $\longrightarrow$ Output `"kuv mus pem roob"` (normalized lowercase, whitespace cleaned, valid tones preserved).
- Input `"NYOB ZOO KOJ PUAS NYOB ZOO"` $\longrightarrow$ Output `"nyob zoo koj puas nyob zoo"`.

### 4.3 chrF++ Scoring Formulation Oracle
For hypothesis $H$ and reference $R$:
$$\text{chrF} = (1 + \beta^2) \frac{\text{chrP} \cdot \text{chrR}}{\beta^2 \text{chrP} + \text{chrR}}, \quad \beta = 2.0$$
chrF++ incorporates character n-grams ($n=1\dots 6$) plus word bigrams ($w=1, 2$). This metric penalizes Hmong tone sandhi deviations significantly less harshly than strict word-BLEU, accurately capturing near-synonymous morphophonemic variants (e.g., *ib tus* vs *ib tug*).

### 4.4 Conversational Real-Time Latency Budget Oracle
The end-to-end latency constraint is strictly:
$$T_{\text{E2E}} = T_{\text{VAD}} + T_{\text{STT}} + T_{\text{MT}} + T_{\text{TTS}} \le 1,500\text{ ms}$$
Theoretical Component Latency Budgets:
- $T_{\text{VAD}}$ (Utterance endpoint detection): $300\text{ ms} - 400\text{ ms}$
- $T_{\text{STT}}$ (Conformer-CTC greedy chunk decoding): $150\text{ ms} - 250\text{ ms}$
- $T_{\text{MT}}$ (NLLB LoRA beam search $\text{width}=4$, $\theta=1.25$): $100\text{ ms} - 200\text{ ms}$
- $T_{\text{TTS}}$ (VITS 2-head 6-layer synthesis + HiFi-GAN): $150\text{ ms} - 250\text{ ms}$
- **Total Expected Latency**: $700\text{ ms} - 1,100\text{ ms} \ll 1,500\text{ ms}$.

---

## 5. Test Suite Structure & Organization

All test files reside in `tests/`:

```
d:/NCKH/tests/
├── __init__.py                     # Package init & shared test fixture loaders
├── test_tier1_feature.py           # Tier 1: Functional Unit & Feature Tests (>=30 tests)
├── test_tier2_boundary.py          # Tier 2: Boundary Value Analysis & Edge Cases
├── test_tier3_combinations.py      # Tier 3: Pairwise Integration & Interface Chaining
└── test_tier4_scenarios.py         # Tier 4: Real-world Scenarios, Latency & Noise Stress
```

### Execution Commands:

1. **Syntax Validation**:
   ```powershell
   python -m py_compile tests/test_*.py
   ```

2. **Full Test Suite Execution**:
   ```powershell
   python -m unittest discover -s tests -p "test_*.py" -v
   ```
   *(or `pytest tests/ -v --tb=short` in container environment)*

3. **Targeted Tier Execution**:
   ```powershell
   python -m unittest tests/test_tier1_feature.py
   python -m unittest tests/test_tier2_boundary.py
   python -m unittest tests/test_tier3_combinations.py
   python -m unittest tests/test_tier4_scenarios.py
   ```

---

## 6. Academic Quality & Forensic Integrity Criteria

To maintain acceptance by top-tier academic venues (e.g., Interspeech, ICASSP, IEEE TASLP):
1. **No Stubs / No Mocks Policy**: No test shall pass conditionally due to an unexercised placeholder. If a model class is instantiated, its actual initialization parameters and forward methods must be invoked.
2. **Deterministic Reproducibility**: Random generators (audio synthetic signals, Gaussian noise) must use a fixed seed (`seed = 42`).
3. **Reproducible LaTeX Output**: Test Tier 4 and academic scripts verify that generated LaTeX tables conform to publication `booktabs` formatting.
