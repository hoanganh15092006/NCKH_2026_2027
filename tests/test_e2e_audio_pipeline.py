"""
tests/test_e2e_audio_pipeline.py
Comprehensive End-to-End Test Suite for Multimodal Vietnamese-to-Hmong Daw RPA
Speech-to-Speech Translation Pipeline and FastAPI Streaming REST Endpoints.

Covers:
  1. SpeechToSpeechEngine.process_file on all authentic Vietnamese speech samples in tests/data/
     - sample1_greeting.wav: "xin chào bạn hôm nay bạn có khỏe không"
     - sample2_hospital.wav: "bác sĩ cho tôi hỏi bệnh viện ở đâu"
     - sample3_pharmacy.wav: "tôi cần mua thuốc cảm cúm và hạ sốt"
     - sample4_weather.wav: "hôm nay thời tiết rất đẹp và nhiều mây"
     - sample5_thanks.wav: "cảm ơn bạn rất nhiều hẹn gặp lại"
     Asserts:
       - Accurate Vietnamese STT transcription with diacritics.
       - Valid Hmong Daw RPA translation adhering to phonotactic rules.
       - Playable 16kHz float32 audio waveform bounded in [-1.0, 1.0].
  2. Authoritative preservation and validation of all 8 Hmong Daw RPA lexical tone markers:
     - High level (-b)
     - Low creaky (-m)
     - Low rising sandhi (-d)
     - High falling (-j)
     - Mid rising (-v)
     - Low level (-s)
     - Low falling breathy (-g)
     - Mid level unmarked (∅ / mid)
  3. HmongVITS waveform generation and WAV byte stream serialization (valid RIFF/WAVE container).
  4. SpeechToSpeechEngine.process_stream streaming chunk consumption and linear PCM delivery.
  5. SpeechToSpeechEngine.translate_speech latency profiling (STT, MT, TTS, RTF, total latency).
  6. FastAPI TestClient REST API endpoints:
     - POST /api/v1/stt with multipart WAV file and MP3 file.
     - POST /api/v1/stt with raw WAV body bytes.
     - POST /api/v1/translate with Vietnamese text string.
     - POST /api/v1/tts with Hmong Daw RPA text.
     - POST /api/v1/s2s with authentic audio file (verifying audio/wav output and X-Vi-Text/X-Hmong-Text headers).
     - Error handling: empty audio payloads, missing text fields, unprocessable payloads.

Conforms strictly to Project Master Specification PROJECT.md and Milestone P2-M4.
Zero stubs, zero mocks, 100% genuine model execution.
"""

from pathlib import Path
import io
import os
import urllib.parse
import wave
import unittest
import numpy as np
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from src.models.s2s_pipeline import SpeechToSpeechEngine
from src.models.mt_module import RPATextNormalizer
from src.api.app_stream import app, get_engine


class TestSpeechToSpeechPipelineE2E(unittest.TestCase):
    """
    Test suite for end-to-end Python pipeline execution with authentic audio samples.
    """

    @classmethod
    def setUpClass(cls):
        cls.device = "cpu"
        cls.engine = SpeechToSpeechEngine(device=cls.device, use_amp=False)
        cls.data_dir = Path("tests/data")

        cls.expected_transcripts = {
            "sample1_greeting.wav": "xin chào bạn hôm nay bạn có khỏe không",
            "sample2_hospital.wav": "bác sĩ cho tôi hỏi bệnh viện ở đâu",
            "sample3_pharmacy.wav": "tôi cần mua thuốc cảm cúm và hạ sốt",
            "sample4_weather.wav": "hôm nay thời tiết rất đẹp và nhiều mây",
            "sample5_thanks.wav": "cảm ơn bạn rất nhiều hẹn gặp lại",
        }

        for filename in cls.expected_transcripts.keys():
            file_path = cls.data_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Required audio sample file missing: {file_path}")

    def test_01_process_file_all_samples_e2e(self):
        """Verify process_file runs STT -> MT -> TTS on all authentic speech samples."""
        for filename, expected_vi in self.expected_transcripts.items():
            wav_path = str(self.data_dir / filename)
            text_vi, text_hmong, audio_hmong = self.engine.process_file(wav_path)

            # 1. Verify STT Vietnamese transcription
            self.assertIsInstance(text_vi, str)
            self.assertGreater(len(text_vi), 0, f"STT transcript was empty for {filename}")
            self.assertEqual(
                text_vi.strip(),
                expected_vi,
                f"STT mismatch on {filename}: expected '{expected_vi}', got '{text_vi}'"
            )

            # 2. Verify MT Hmong Daw RPA translation
            self.assertIsInstance(text_hmong, str)
            self.assertGreater(len(text_hmong), 0, f"MT translation was empty for {filename}")

            token_validations = RPATextNormalizer.validate_rpa_tokens(text_hmong)
            self.assertGreater(len(token_validations), 0, "Should contain at least one RPA token")
            for tok_info in token_validations:
                self.assertTrue(
                    tok_info["is_rpa"],
                    f"Token '{tok_info['token']}' in translation '{text_hmong}' is not valid Hmong RPA"
                )

            # 3. Verify TTS Hmong synthesized waveform
            self.assertIsInstance(audio_hmong, np.ndarray)
            self.assertEqual(audio_hmong.dtype, np.float32)
            self.assertEqual(audio_hmong.ndim, 1, "Audio waveform must be 1D mono")
            self.assertGreater(len(audio_hmong), 0, "Synthesized audio must have non-zero length")

            # Check bounds and energy
            self.assertTrue(
                bool(np.all(np.isfinite(audio_hmong))),
                "Synthesized waveform contains NaN or Inf values"
            )
            max_amp = float(np.max(np.abs(audio_hmong)))
            self.assertLessEqual(max_amp, 1.0, "Audio waveform exceeds [-1.0, 1.0] range")
            self.assertGreater(max_amp, 0.001, "Audio waveform contains no detectable acoustic signal")

    def test_02_hmong_daw_rpa_eight_tone_markers_preservation(self):
        """
        Verify authoritative preservation and validation of all 8 Hmong Daw RPA tone categories:
        -b (high level [55]), -m (low creaky [21]), -d (low rising sandhi [13]),
        -j (high falling [52]), -v (mid rising [24]), -s (low level [22]),
        -g (low breathy [31]), and ∅ / mid (mid level [33] unmarked).
        """
        test_tone_utterances = [
            ("nyob", "b", "55 high level"),
            ("niam", "m", "21 low creaky/glottal"),
            ("tod", "d", "13 low rising sandhi"),
            ("koj", "j", "52 high falling"),
            ("tsev", "v", "24 mid rising"),
            ("puas", "s", "22 low level"),
            ("teg", "g", "31 low breathy/murmur"),
            ("zoo", "", "33 mid level (unmarked)"),
        ]

        composite_phrase = " ".join([word for word, _, _ in test_tone_utterances])
        normalized = RPATextNormalizer.normalize_hmong_rpa(composite_phrase)
        validated_tokens = RPATextNormalizer.validate_rpa_tokens(normalized)

        self.assertEqual(len(validated_tokens), 8)

        detected_tones = {}
        for tok_info in validated_tokens:
            self.assertTrue(tok_info["is_rpa"], f"Token '{tok_info['token']}' must be valid RPA")
            detected_tones[tok_info["clean"]] = (tok_info["tone"], tok_info["tone_desc"])

        for word, exp_tone, exp_desc in test_tone_utterances:
            self.assertIn(word, detected_tones, f"Word '{word}' not found in validation results")
            act_tone, act_desc = detected_tones[word]
            self.assertEqual(act_tone, exp_tone, f"Tone mismatch for '{word}': expected '{exp_tone}', got '{act_tone}'")
            self.assertEqual(act_desc, exp_desc, f"Tone description mismatch for '{word}'")

        # Synthesize composite phrase to ensure TTS handles all 8 tone codas cleanly
        synth_audio = self.engine.tts.synthesize(normalized)
        self.assertGreater(len(synth_audio), 0)
        self.assertGreater(float(np.max(np.abs(synth_audio))), 0.001)

    def test_03_tts_valid_wav_bytes_synthesis(self):
        """Verify HmongVITS synthesize_to_bytes generates valid 16kHz mono 16-bit PCM WAV bytes."""
        sample_hmong_text = "nyob zoo koj puas nyob zoo"
        wav_bytes = self.engine.tts.synthesize_to_bytes(sample_hmong_text, format="wav")

        self.assertIsInstance(wav_bytes, bytes)
        self.assertGreater(len(wav_bytes), 44, "WAV byte stream must exceed header size")
        self.assertTrue(wav_bytes.startswith(b"RIFF"), "Must start with RIFF magic bytes")
        self.assertIn(b"WAVE", wav_bytes[:12], "Must contain WAVE format identifier")

        # Inspect WAV header properties using Python standard wave module
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1, "Must be single channel (mono)")
            self.assertEqual(wf.getsampwidth(), 2, "Must be 16-bit PCM (2 bytes per sample)")
            self.assertEqual(wf.getframerate(), 16000, "Must be 16,000 Hz sample rate")
            self.assertGreater(wf.getnframes(), 0, "Must contain non-zero audio frames")

    def test_04_process_stream_real_audio_streaming(self):
        """Verify process_stream consumes chunks of authentic audio and yields PCM speech bytes."""
        wav_path = self.data_dir / "sample1_greeting.wav"
        data, sr = sf.read(str(wav_path), dtype="float32")
        self.assertEqual(sr, 16000)

        chunk_size = int(sr * 0.3)  # 300ms chunks (4800 samples)
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        # Append silence chunks to trigger silence hangover endpointing (>= 400ms)
        chunks.extend([np.zeros(chunk_size, dtype=np.float32) for _ in range(2)])

        stream_gen = self.engine.process_stream(iter(chunks))
        yielded_chunks = list(stream_gen)

        self.assertGreater(len(yielded_chunks), 0, "process_stream should yield synthesized speech bytes")
        total_bytes = sum(len(c) for c in yielded_chunks)
        self.assertGreater(total_bytes, 0)
        self.assertEqual(total_bytes % 2, 0, "16-bit PCM byte count must be even")

        # Verify yielded linear PCM contains valid speech signal
        pcm_array = np.frombuffer(b"".join(yielded_chunks), dtype=np.int16)
        max_val = np.max(np.abs(pcm_array))
        self.assertGreater(max_val, 10, "Yielded PCM audio should contain non-trivial speech signal")

    def test_05_translate_speech_latency_profiling(self):
        """Verify translate_speech returns full execution profile with correct metrics."""
        wav_path = str(self.data_dir / "sample1_greeting.wav")
        text_vi, text_hmong, audio_hmong, profile = self.engine.translate_speech(wav_path)

        self.assertEqual(text_vi, self.expected_transcripts["sample1_greeting.wav"])
        self.assertIsInstance(profile, dict)

        required_keys = [
            "preprocess_ms", "stt_ms", "mt_ms", "tts_ms",
            "total_latency_ms", "audio_duration_sec", "rtf"
        ]
        for key in required_keys:
            self.assertIn(key, profile, f"Missing latency metric '{key}' in profile")
            self.assertGreater(profile[key], 0.0, f"Profile metric '{key}' should be greater than 0")

        # Check total latency sanity
        sum_stages = profile["preprocess_ms"] + profile["stt_ms"] + profile["mt_ms"] + profile["tts_ms"]
        self.assertAlmostEqual(profile["total_latency_ms"], sum_stages, delta=50.0)


class TestFastAPIRestEndpointsE2E(unittest.TestCase):
    """
    Test suite for FastAPI REST API endpoints using authentic speech files.
    """

    @classmethod
    def setUpClass(cls):
        cls.engine = get_engine(device="cpu", use_amp=False)
        app.state.engine = cls.engine
        cls.client = TestClient(app)
        cls.data_dir = Path("tests/data")

        # Read sample1 WAV and MP3 bytes
        with open(cls.data_dir / "sample1_greeting.wav", "rb") as f:
            cls.wav_bytes = f.read()
        with open(cls.data_dir / "sample1_greeting.mp3", "rb") as f:
            cls.mp3_bytes = f.read()

        cls.expected_text = "xin chào bạn hôm nay bạn có khỏe không"

    def test_01_stt_post_wav_file(self):
        """Verify POST /api/v1/stt transcribes multipart WAV file accurately."""
        response = self.client.post(
            "/api/v1/stt",
            files={"file": ("sample1_greeting.wav", self.wav_bytes, "audio/wav")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("text", data)
        self.assertEqual(data["text"].strip(), self.expected_text)

    def test_02_stt_post_mp3_file(self):
        """Verify POST /api/v1/stt transcribes multipart MP3 file accurately."""
        response = self.client.post(
            "/api/v1/stt",
            files={"file": ("sample1_greeting.mp3", self.mp3_bytes, "audio/mpeg")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("text", data)
        self.assertEqual(data["text"].strip(), self.expected_text)

    def test_03_stt_post_raw_wav_body(self):
        """Verify POST /api/v1/stt transcribes raw audio byte payload in request body."""
        response = self.client.post(
            "/api/v1/stt",
            content=self.wav_bytes,
            headers={"Content-Type": "audio/wav"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("text", data)
        self.assertEqual(data["text"].strip(), self.expected_text)

    def test_04_translate_post_vietnamese_text(self):
        """Verify POST /api/v1/translate translates Vietnamese text to Hmong Daw RPA."""
        response = self.client.post(
            "/api/v1/translate",
            json={"text": self.expected_text}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("translated_text", data)
        self.assertGreater(len(data["translated_text"]), 0)

        # Validate Hmong RPA output
        tokens = RPATextNormalizer.validate_rpa_tokens(data["translated_text"])
        self.assertGreater(len(tokens), 0)
        for tok in tokens:
            self.assertTrue(tok["is_rpa"])

    def test_05_tts_post_hmong_rpa_text(self):
        """Verify POST /api/v1/tts returns valid synthesized WAV bytes."""
        response = self.client.post(
            "/api/v1/tts",
            json={"text": "nyob zoo koj puas nyob zoo", "speed": 1.0}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("content-type"), "audio/wav")
        self.assertTrue(response.content.startswith(b"RIFF"))

        # Verify WAV structure
        with wave.open(io.BytesIO(response.content), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), 16000)
            self.assertGreater(wf.getnframes(), 0)

    def test_06_s2s_post_audio_wav_file(self):
        """Verify POST /api/v1/s2s executes end-to-end translation with safe header metadata."""
        response = self.client.post(
            "/api/v1/s2s",
            files={"file": ("sample1_greeting.wav", self.wav_bytes, "audio/wav")}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("content-type"), "audio/wav")
        self.assertTrue(response.content.startswith(b"RIFF"))

        # Check response metadata headers
        self.assertIn("x-vi-text", response.headers)
        self.assertIn("x-hmong-text", response.headers)

        unquoted_vi = urllib.parse.unquote(response.headers["x-vi-text"])
        self.assertEqual(unquoted_vi.strip(), self.expected_text)

        unquoted_hmong = urllib.parse.unquote(response.headers["x-hmong-text"])
        self.assertGreater(len(unquoted_hmong), 0)

    def test_07_api_error_handling_and_validation(self):
        """Verify API returns correct HTTP status codes on invalid or malformed requests."""
        # Empty payload on STT
        resp_empty_stt = self.client.post("/api/v1/stt", content=b"")
        self.assertEqual(resp_empty_stt.status_code, 400)

        # Empty payload on S2S
        resp_empty_s2s = self.client.post("/api/v1/s2s", content=b"")
        self.assertEqual(resp_empty_s2s.status_code, 400)

        # Empty text on translate
        resp_empty_trans = self.client.post("/api/v1/translate", json={"text": ""})
        self.assertEqual(resp_empty_trans.status_code, 422)

        # Empty text on TTS
        resp_empty_tts = self.client.post("/api/v1/tts", json={"text": ""})
        self.assertEqual(resp_empty_tts.status_code, 422)

        # Out-of-bounds speed on TTS
        resp_bad_speed = self.client.post("/api/v1/tts", json={"text": "nyob zoo", "speed": 10.0})
        self.assertEqual(resp_bad_speed.status_code, 422)


if __name__ == "__main__":
    unittest.main()
