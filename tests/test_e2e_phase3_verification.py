"""
tests/test_e2e_phase3_verification.py
Authoritative End-to-End Verification Test Suite for Phase 3 (Milestone P3-M4).

Verifies ALL Acceptance Criteria across Requirements R1, R2, R3, and Verification:
  1. Dual Input Verification:
     - Direct text input via POST /api/v1/t2s and POST /api/v1/s2s (bypassing STT).
     - Audio file upload (WAV and MP3) processing through STT -> MT -> TTS.
     - Web UI elements in INDEX_HTML (#viTextInput, #translateTextBtn, #clearTextBtn,
       #nativeAudioPlayer, #autoplayNotice, and autoplay mitigation JavaScript).
  2. Translation Coverage & Tone Integrity:
     - Multi-domain conversational Vietnamese sentences translate with zero <unk> tokens.
     - All 8 White H'Mong RPA lexical tones preserved (-b, -m, -d, -j, -v, -s, -g, mid).
     - Deterministic Number Synthesizer converts Arabic digits and Vietnamese words to RPA.
     - Hyphenated compound words (e.g. me-nyuam) recognized and validated.
  3. TTS Waveform & Browser Playback:
     - Synthesized audio RMS amplitude strictly within [-20.0, -16.0] dBFS.
     - Zero DC offset (< 1e-3) and peak amplitude <= 0.99 (no clipping).
     - Standard 44-byte RIFF/WAV headers validated by wave and soundfile.
  4. Real-World YouTube Speech Verification:
     - Authentic Vietnamese broadcast speech from YouTube (https://www.youtube.com/watch?v=SDA4uB_1gSY).
     - 3 distinct 16kHz mono WAV fixtures in tests/data/yt_speech/.
     - End-to-end STT transcript accuracy and synthesized H'Mong audio generation.
"""

import io
import os
import sys
import unittest
import wave
import numpy as np
import soundfile as sf
import torch
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.app_stream import app, get_engine, INDEX_HTML
from src.models.mt_module import (
    HmongTranslationModel,
    RPATextNormalizer,
    HmongNumberSynthesizer
)
from src.models.tts_module import HmongVITS
from src.models.stt_module import VietnameseSTT


# ===========================================================================
# 1. Dual Input Verification Suite (Requirement R1 & Acceptance Criteria)
# ===========================================================================
class TestPhase3DualInputVerification(unittest.TestCase):
    """Verifies direct text input (T2S and dual S2S), audio upload (WAV/MP3), and Web UI controls."""

    @classmethod
    def setUpClass(cls):
        cls.engine = get_engine(device="cpu", use_amp=False)
        app.state.engine = cls.engine
        cls.client = TestClient(app)

    def test_01_direct_text_input_t2s_endpoint(self):
        """Verify dedicated POST /api/v1/t2s translates text and returns valid WAV without audio."""
        test_payloads = [
            {"text": "xin chào bạn", "speed": 1.0},
            {"text": "bác sĩ cho tôi hỏi bệnh viện ở đâu", "speed": 1.0},
            {"text": "tôi cần mua thuốc cảm cúm", "speed": 1.2},
        ]
        for payload in test_payloads:
            res = self.client.post("/api/v1/t2s", json=payload)
            self.assertEqual(res.status_code, 200, f"Failed on text: {payload['text']}")
            self.assertEqual(res.headers.get("content-type"), "audio/wav")
            self.assertTrue(res.content.startswith(b"RIFF"), "Output does not start with RIFF header")
            self.assertIn("x-vi-text", res.headers)
            self.assertIn("x-hmong-text", res.headers)

            # Verify WAV content is decodable
            data, sr = sf.read(io.BytesIO(res.content))
            self.assertEqual(sr, 16000)
            self.assertGreater(len(data), 0)

        # Boundary checks: Empty and blank text must return 400 or 422
        res_empty = self.client.post("/api/v1/t2s", json={"text": ""})
        self.assertIn(res_empty.status_code, [400, 422])
        res_blank = self.client.post("/api/v1/t2s", json={"text": "   "})
        self.assertIn(res_blank.status_code, [400, 422])

    def test_02_direct_text_input_s2s_dual_endpoint(self):
        """Verify POST /api/v1/s2s accepts direct text (both JSON and Form data) bypassing STT."""
        # 1. JSON payload
        res_json = self.client.post("/api/v1/s2s", json={"text": "hôm nay thời tiết thế nào"})
        self.assertEqual(res_json.status_code, 200)
        self.assertTrue(res_json.content.startswith(b"RIFF"))
        self.assertIn("x-vi-text", res_json.headers)
        self.assertIn("x-hmong-text", res_json.headers)
        data, sr = sf.read(io.BytesIO(res_json.content))
        self.assertEqual(sr, 16000)
        self.assertGreater(len(data), 0)

        # 2. Form data payload
        res_form = self.client.post("/api/v1/s2s", data={"text": "chúc mừng năm mới"})
        self.assertEqual(res_form.status_code, 200)
        self.assertTrue(res_form.content.startswith(b"RIFF"))

        # 3. Reject empty request when neither text nor audio is supplied
        res_empty = self.client.post("/api/v1/s2s", json={})
        self.assertEqual(res_empty.status_code, 400)

    def test_03_audio_file_upload_wav_and_mp3(self):
        """Verify POST /api/v1/s2s processes uploaded audio files in both WAV and MP3 formats."""
        wav_path = os.path.join(PROJECT_ROOT, "tests", "data", "sample1_greeting.wav")
        mp3_path = os.path.join(PROJECT_ROOT, "tests", "data", "sample1_greeting.mp3")

        self.assertTrue(os.path.exists(wav_path), f"Fixture not found: {wav_path}")
        self.assertTrue(os.path.exists(mp3_path), f"Fixture not found: {mp3_path}")

        # 1. Upload WAV
        with open(wav_path, "rb") as f:
            res_wav = self.client.post(
                "/api/v1/s2s",
                files={"file": ("sample1_greeting.wav", f, "audio/wav")}
            )
        self.assertEqual(res_wav.status_code, 200)
        self.assertTrue(res_wav.content.startswith(b"RIFF"))
        self.assertIn("x-vi-text", res_wav.headers)
        self.assertIn("x-hmong-text", res_wav.headers)

        # 2. Upload MP3
        with open(mp3_path, "rb") as f:
            res_mp3 = self.client.post(
                "/api/v1/s2s",
                files={"file": ("sample1_greeting.mp3", f, "audio/mpeg")}
            )
        self.assertEqual(res_mp3.status_code, 200)
        self.assertTrue(res_mp3.content.startswith(b"RIFF"))
        self.assertIn("x-vi-text", res_mp3.headers)
        self.assertIn("x-hmong-text", res_mp3.headers)

    def test_04_web_ui_text_input_and_playback_elements(self):
        """Verify Web UI contains all required text input elements and autoplay mitigation controls."""
        # 1. Text input textarea and buttons
        self.assertIn('id="viTextInput"', INDEX_HTML)
        self.assertIn('id="translateTextBtn"', INDEX_HTML)
        self.assertIn('id="clearTextBtn"', INDEX_HTML)

        # 2. Native audio controls and autoplay notification banner
        self.assertIn('id="nativeAudioPlayer"', INDEX_HTML)
        self.assertIn('id="autoplayNotice"', INDEX_HTML)
        self.assertIn('controls', INDEX_HTML)

        # 3. Client JavaScript bindings
        self.assertIn('triggerTextTranslation', INDEX_HTML)
        self.assertIn('unlockAudioPlayback', INDEX_HTML)
        self.assertIn('/api/v1/t2s', INDEX_HTML)
        self.assertIn('btn-replay-pulse', INDEX_HTML)


# ===========================================================================
# 2. Translation Coverage & Tone Integrity Suite (Requirement R2)
# ===========================================================================
class TestPhase3TranslationAndToneIntegrity(unittest.TestCase):
    """Verifies zero <unk> tokens, 8-tone preservation, number synthesizer, and hyphenated compounds."""

    @classmethod
    def setUpClass(cls):
        cls.mt = HmongTranslationModel()

    def test_01_multi_domain_conversational_zero_unk(self):
        """Verify common conversational sentences across multiple daily domains emit zero <unk> tokens."""
        domain_sentences = [
            # Greetings & Courtesy
            "xin chào",
            "cảm ơn bạn",
            "tạm biệt",
            "hẹn gặp lại",
            "rất vui được gặp bạn",
            "xin lỗi",
            # Interrogatives & Directions
            "bây giờ là mấy giờ",
            "giá bao nhiêu",
            "đường đi chợ ở đâu",
            "đi thẳng rồi rẽ phải",
            "nhà bạn ở đâu",
            # Daily life, Food & Commerce
            "tôi muốn mua thịt lợn",
            "cho tôi một cốc nước",
            "tôi muốn ăn cơm",
            "tôi rất đói bụng",
            # Healthcare & Emergency
            "tôi cần mua thuốc",
            "tôi bị đau đầu và sốt cao",
            "xin hãy gọi bác sĩ giúp tôi",
            "bệnh viện ở đâu",
            # Weather & Time
            "hôm nay trời mưa rất to",
            "ngày mai trời nắng đẹp",
            "thời tiết hôm nay rất lạnh",
        ]
        for sent in domain_sentences:
            translation = self.mt.translate(sent)
            self.assertNotIn(
                "<unk>", translation.lower(),
                f"<unk> emitted in translation of '{sent}' -> '{translation}'"
            )
            self.assertGreater(
                len(translation.strip()), 0,
                f"Translation for '{sent}' is unexpectedly empty"
            )

    def test_02_all_eight_white_hmong_tones_preservation(self):
        """Verify all 8 White H'Mong RPA lexical tones are recognized and preserved."""
        # 8 Lexical Tones in White Hmong RPA:
        # -b (high), -m (low-glottalized), -d (low-rising), -j (high-falling),
        # -v (mid-rising), -s (low), -g (breathy/creaky), and unmarked mid tone.
        tone_test_words = {
            "b": "nyob",
            "m": "kaum",
            "d": "tod",
            "j": "dej",
            "v": "kuv",
            "s": "mus",
            "g": "tsaug",
            "": "zoo"  # unmarked mid tone
        }
        for expected_tone, word in tone_test_words.items():
            normalized = RPATextNormalizer.normalize_hmong_rpa(f"nws pom {word} .")
            token_validation = RPATextNormalizer.validate_rpa_tokens(normalized)
            matching = [t for t in token_validation if t["clean"] == word]
            self.assertEqual(len(matching), 1, f"Word '{word}' was not found in validation output")
            self.assertTrue(matching[0]["is_rpa"], f"Word '{word}' failed RPA regex validation")
            self.assertEqual(
                matching[0]["tone"], expected_tone,
                f"Expected tone '{expected_tone}' for word '{word}', got '{matching[0]['tone']}'"
            )

    def test_03_number_synthesizer_arabic_and_vietnamese_numerals(self):
        """Verify HmongNumberSynthesizer accurately converts Arabic digits and Vietnamese text to valid RPA."""
        # 1. Arabic digits conversion
        digit_cases = [
            (0, "zeej"),
            (1, "ib"),
            (5, "tsib"),
            (10, "kaum"),
            (15, "kaum tsib"),
            (20, "nees nkaum"),
            (25, "nees nkaum tsib"),
            (150, "ib puas tsib caug"),
            (2026, "ob txhiab nees nkaum rau"),
            (1000000, "ib lab"),
        ]
        for num, expected_rpa in digit_cases:
            result = HmongNumberSynthesizer.int_to_hmong_rpa(num)
            self.assertEqual(result, expected_rpa)
            val = RPATextNormalizer.validate_rpa_tokens(result)
            self.assertTrue(all(v["is_rpa"] for v in val), f"Invalid RPA in number {num}: {result}")

        # 2. Vietnamese numeral words in MT translation
        vi_numeral_sentences = [
            ("một hai ba bốn năm", "ib ob peb plaub tsib"),
            ("mười lăm", "kaum tsib"),
            ("hai mươi lăm", "nees nkaum tsib"),
            ("một trăm năm mươi", "ib puas tsib caug"),
        ]
        for vi_text, expected_hmong in vi_numeral_sentences:
            trans = self.mt.translate(vi_text)
            self.assertNotIn("<unk>", trans)
            self.assertEqual(trans, expected_hmong)

    def test_04_hyphenated_compound_words_validation(self):
        """Verify RPATextNormalizer validates hyphenated compound words like 'me-nyuam' without errors."""
        compounds = ["me-nyuam", "poj-niam", "txiv-neej", "kwv-tij"]
        for compound in compounds:
            val = RPATextNormalizer.validate_rpa_tokens(compound)
            self.assertEqual(len(val), 1, f"Expected 1 compound token for '{compound}'")
            self.assertTrue(
                val[0]["is_rpa"],
                f"Hyphenated compound '{compound}' was not recognized as valid RPA"
            )


# ===========================================================================
# 3. TTS Waveform & Browser Playback Suite (Requirement R3)
# ===========================================================================
class TestPhase3TTSWaveformAndPlayback(unittest.TestCase):
    """Verifies TTS RMS amplitude in corridor [-20, -16] dBFS, zero DC offset, and 44-byte RIFF headers."""

    @classmethod
    def setUpClass(cls):
        cls.tts = HmongVITS(sample_rate=16000, device="cpu")
        cls.sample_phrases = [
            "nyob zoo",
            "ua tsaug",
            "kuv nyob zoo",
            "sib ntsib dua",
            "koj tuaj qhov twg tuaj",
            "hmoob dawb",
            "hnub no huab cua zoo heev",
        ]

    def test_01_rms_amplitude_strictly_in_target_corridor(self):
        """Verify synthesized waveforms achieve RMS amplitude strictly between -20.0 dBFS and -16.0 dBFS."""
        for phrase in self.sample_phrases:
            audio = self.tts.synthesize(phrase)
            self.assertIsInstance(audio, np.ndarray)
            self.assertEqual(audio.dtype, np.float32)
            self.assertGreater(len(audio), 0)

            rms = float(np.sqrt(np.mean(audio ** 2)))
            dbfs = 20.0 * np.log10(rms) if rms > 0 else -999.0

            self.assertGreaterEqual(
                dbfs, -20.0,
                f"Audio for '{phrase}' too quiet: {dbfs:.2f} dBFS < -20.0 dBFS"
            )
            self.assertLessEqual(
                dbfs, -16.0,
                f"Audio for '{phrase}' too loud: {dbfs:.2f} dBFS > -16.0 dBFS"
            )

    def test_02_dc_offset_elimination_and_no_clipping(self):
        """Verify zero DC offset (< 1e-3) and peak amplitude <= 0.99 with no digital clipping."""
        for phrase in self.sample_phrases:
            audio = self.tts.synthesize(phrase)
            dc = float(np.mean(audio))
            peak = float(np.max(np.abs(audio)))

            self.assertLess(
                abs(dc), 1e-3,
                f"Audio for '{phrase}' has excessive DC offset: {dc:.6f} >= 1e-3"
            )
            self.assertLessEqual(
                peak, 0.99,
                f"Audio for '{phrase}' exceeded peak safety ceiling: {peak:.4f} > 0.99"
            )

    def test_03_standard_compliant_riff_wav_header(self):
        """Verify synthesize_to_bytes produces 100% compliant 44-byte RIFF/WAV headers validated by wave & soundfile."""
        for phrase in self.sample_phrases:
            wav_bytes = self.tts.synthesize_to_bytes(phrase, format="wav")
            self.assertIsInstance(wav_bytes, bytes)
            self.assertGreater(len(wav_bytes), 44)

            # 1. Binary header structure inspection
            self.assertEqual(wav_bytes[:4], b"RIFF")
            self.assertEqual(wav_bytes[8:12], b"WAVE")
            self.assertEqual(wav_bytes[12:16], b"fmt ")

            # 2. wave module validation
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1, "Must be mono audio")
                self.assertEqual(wf.getsampwidth(), 2, "Must be 16-bit linear PCM")
                self.assertEqual(wf.getframerate(), 16000, "Must be 16000 Hz")
                self.assertGreater(wf.getnframes(), 0)

            # 3. soundfile validation
            data, sr = sf.read(io.BytesIO(wav_bytes))
            self.assertEqual(sr, 16000)
            self.assertGreater(len(data), 0)

    def test_04_empty_and_whitespace_input_handling(self):
        """Verify empty and whitespace inputs return empty audio without throwing exceptions."""
        for blank in ["", "   ", "\t\n"]:
            audio = self.tts.synthesize(blank)
            self.assertEqual(len(audio), 0)
            pcm_bytes = self.tts.synthesize_to_bytes(blank, format="pcm")
            self.assertEqual(len(pcm_bytes), 0)


# ===========================================================================
# 4. Real-World YouTube Speech Verification Suite
# ===========================================================================
class TestPhase3RealWorldYouTubeSpeechVerification(unittest.TestCase):
    """Verifies authentic Vietnamese broadcast speech segments extracted from YouTube."""

    @classmethod
    def setUpClass(cls):
        cls.stt = VietnameseSTT(device="cpu")
        cls.mt = HmongTranslationModel()
        cls.tts = HmongVITS(sample_rate=16000, device="cpu")
        cls.engine = get_engine(device="cpu", use_amp=False)
        app.state.engine = cls.engine
        cls.client = TestClient(app)

        cls.yt_dir = os.path.join(PROJECT_ROOT, "tests", "data", "yt_speech")
        cls.sample_files = [
            os.path.join(cls.yt_dir, "yt_sample_01.wav"),
            os.path.join(cls.yt_dir, "yt_sample_02.wav"),
            os.path.join(cls.yt_dir, "yt_sample_03.wav"),
        ]

    def test_01_youtube_audio_fixtures_exist_and_valid(self):
        """Verify all 3 authentic YouTube speech WAV fixtures exist and conform to 16kHz mono PCM16."""
        self.assertTrue(os.path.isdir(self.yt_dir), f"Directory {self.yt_dir} does not exist")

        for sample_path in self.sample_files:
            self.assertTrue(
                os.path.exists(sample_path),
                f"Required YouTube speech fixture missing: {sample_path}"
            )
            info = sf.info(sample_path)
            self.assertEqual(info.samplerate, 16000, f"Sample rate must be 16000 Hz, got {info.samplerate}")
            self.assertEqual(info.channels, 1, f"Audio must be mono, got {info.channels}")
            self.assertEqual(info.subtype, "PCM_16", f"Subtype must be PCM_16, got {info.subtype}")
            self.assertGreater(info.duration, 2.0, f"Duration too short: {info.duration}s")
            self.assertLess(info.duration, 10.0, f"Duration too long: {info.duration}s")

    def test_02_youtube_speech_stt_transcription_accuracy(self):
        """Verify VietnameseSTT transcribes authentic YouTube speech segments with high keyword accuracy."""
        expected_keywords = {
            "yt_sample_01.wav": ["sơn la", "tuần tra", "kiểm soát", "đường", "cảnh sát", "giao thông"],
            "yt_sample_02.wav": ["công an", "tỉnh", "bắt giữ", "đối tượng", "ma túy"],
            "yt_sample_03.wav": ["cán bộ", "bệnh viện", "thăm khám", "điều trị", "vụ việc", "chức năng"]
        }

        for sample_path in self.sample_files:
            fname = os.path.basename(sample_path)
            transcription = self.stt.transcribe_file(sample_path)
            self.assertIsInstance(transcription, str)
            self.assertGreater(len(transcription.strip()), 0, f"Empty transcription for {fname}")

            # Verify presence of expected broadcast domain keywords
            kw_list = expected_keywords[fname]
            matched_kws = [kw for kw in kw_list if kw in transcription.lower()]
            self.assertGreaterEqual(
                len(matched_kws), 2,
                f"STT transcription for {fname} ('{transcription}') matched fewer than 2 keywords from {kw_list}"
            )

    def test_03_youtube_end_to_end_pipeline_speech_to_speech(self):
        """Verify full cascaded pipeline (STT -> MT -> TTS) on authentic YouTube audio fixtures."""
        for sample_path in self.sample_files:
            fname = os.path.basename(sample_path)

            # 1. STT: Audio -> Vietnamese Text
            transcription = self.stt.transcribe_file(sample_path)
            self.assertGreater(len(transcription.strip()), 0)

            # 2. MT: Vietnamese Text -> White Hmong Daw RPA
            translation = self.mt.translate(transcription)
            self.assertNotIn("<unk>", translation.lower(), f"<unk> emitted on {fname}")
            self.assertGreater(len(translation.strip()), 0)

            # 3. TTS: Hmong RPA -> Synthesized Speech Audio
            audio_out = self.tts.synthesize(translation)
            self.assertIsInstance(audio_out, np.ndarray)
            self.assertGreater(len(audio_out), 0, f"TTS produced empty audio for {fname}")
            self.assertEqual(audio_out.dtype, np.float32)

            # Audio sanity checks
            peak = float(np.max(np.abs(audio_out)))
            dc = float(np.mean(audio_out))
            self.assertLessEqual(peak, 0.99)
            self.assertLess(abs(dc), 1e-3)

    def test_04_youtube_samples_via_api_s2s_endpoint(self):
        """Verify authentic YouTube audio fixtures post to POST /api/v1/s2s and return valid WAV audio."""
        for sample_path in self.sample_files:
            fname = os.path.basename(sample_path)
            with open(sample_path, "rb") as f:
                res = self.client.post(
                    "/api/v1/s2s",
                    files={"file": (fname, f, "audio/wav")}
                )

            self.assertEqual(res.status_code, 200, f"POST /api/v1/s2s failed on {fname}: {res.text}")
            self.assertEqual(res.headers.get("content-type"), "audio/wav")
            self.assertTrue(res.content.startswith(b"RIFF"))
            self.assertIn("x-vi-text", res.headers)
            self.assertIn("x-hmong-text", res.headers)

            # Verify synthesized WAV output decodes cleanly
            data, sr = sf.read(io.BytesIO(res.content))
            self.assertEqual(sr, 16000)
            self.assertGreater(len(data), 0)


if __name__ == "__main__":
    unittest.main()
