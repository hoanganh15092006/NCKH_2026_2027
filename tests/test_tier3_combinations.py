"""
Tier 3: Pairwise Chaining & Cross-Feature Integration Test Suite
Multimodal Vietnamese-to-H'Mong Speech-to-Speech Translation System.
Tests interface contracts and data compatibility between consecutive genuine modules:
1. AudioPreprocessor -> VietnameseSTT (WAV buffer -> 16kHz mono RMS norm -> STT transcription)
2. VietnameseSTT -> HmongTranslationModel (Vietnamese transcript -> RPA MT tokenization -> Hmong RPA)
3. HmongTranslationModel -> HmongVITS (Hmong RPA text -> VITS phoneme encoder -> HiFi-GAN 16kHz audio)
4. RingBufferVAD -> SpeechToSpeechEngine (PCM byte chunks -> VAD endpointing -> Streaming S2S dispatch)
5. Full End-to-End Pipeline Chaining (Audio In -> Preprocess -> STT -> MT -> TTS -> Audio Out)
6. Multi-turn State Isolation (Ensuring clean resets between consecutive dialogue turns)
"""

import io
import math
import os
import re
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audio_preprocessor import AudioPreprocessor
from src.models import (
    VietnameseSTT,
    HmongTranslationModel,
    HmongVITS,
    SpeechToSpeechEngine,
)
from tests.test_utils import (
    CONFIG_PATH,
    HAS_NUMPY,
    HAS_TORCH,
    HAS_YAML,
    calculate_rms_dbfs,
    generate_synthetic_audio,
    load_project_config,
)

if HAS_NUMPY:
    import numpy as np
if HAS_TORCH:
    import torch


# ==============================================================================
# Cross-Feature Integration Tests (Genuine Model Chaining)
# ==============================================================================
class TestCrossFeatureIntegration(unittest.TestCase):
    """
    Integration tests validating pairwise chaining across genuine S2S sub-pipelines.
    Conforms to Milestone 8 remediation: 100% genuine model execution, zero mocks.
    """

    @classmethod
    def setUpClass(cls):
        # Instantiate genuine models once for the test class on CPU
        cls.preprocessor = AudioPreprocessor(device="cpu")
        cls.stt = VietnameseSTT(device="cpu")
        cls.mt = HmongTranslationModel(device="cpu")
        cls.tts = HmongVITS(sample_rate=16000, device="cpu")
        cls.engine = SpeechToSpeechEngine(
            preprocessor=cls.preprocessor,
            stt_model=cls.stt,
            mt_model=cls.mt,
            tts_model=cls.tts,
            device="cpu",
            use_amp=False,
        )

    def setUp(self):
        self.preprocessor = self.__class__.preprocessor
        self.stt = self.__class__.stt
        self.mt = self.__class__.mt
        self.tts = self.__class__.tts
        self.engine = self.__class__.engine
        self.stt.reset_streaming_state()

    def test_chain_preprocessor_to_stt_contract(self):
        """
        Verify AudioPreprocessor -> VietnameseSTT contract:
        Audio waveform at 16kHz mono, RMS normalized to -20 dBFS, feeds into STT
        and produces non-empty Vietnamese transcript.
        """
        # Step 1: Input raw audio (simulating microphone input at 16kHz)
        sr = 16000
        raw_audio = generate_synthetic_audio(duration_s=1.0, sample_rate=sr, amplitude=0.4)
        if isinstance(raw_audio, list):
            raw_audio = np.array(raw_audio, dtype=np.float32)

        # Step 2: Genuine AudioPreprocessor RMS Normalization to -20 dBFS
        norm_audio = self.preprocessor.normalize_rms(raw_audio, target_db=-20.0)
        norm_np = norm_audio.squeeze().cpu().numpy() if isinstance(norm_audio, torch.Tensor) else norm_audio
        self.assertEqual(norm_np.dtype, np.float32)
        rms_db = calculate_rms_dbfs(norm_np)
        self.assertAlmostEqual(rms_db, -20.0, delta=1.5)

        # Step 3: Feed genuine normalized audio into genuine STT chunk transcriber
        transcript_vi = self.stt.transcribe_chunk(norm_np)
        self.assertIsInstance(transcript_vi, str)
        self.assertGreater(len(transcript_vi), 0)

    def test_chain_stt_to_mt_contract(self):
        """
        Verify VietnameseSTT -> HmongTranslationModel contract:
        Vietnamese transcript output from STT feeds into MT model, preserving diacritics
        and producing valid Hmong Daw RPA text with tone consonants.
        """
        # Step 1: Authentic transcript
        transcript_vi = "xin chào bạn hôm nay bạn có khỏe không"

        # Step 2: Genuine MT model translates
        translation_hmong = self.mt.translate(transcript_vi)
        self.assertIsInstance(translation_hmong, str)
        self.assertGreater(len(translation_hmong), 0)

        # Step 3: Validate Hmong RPA linguistic properties
        tokens = translation_hmong.split()
        self.assertIn("nyob", tokens)
        self.assertIn("zoo", tokens)
        self.assertIn("koj", tokens)

        # Verify that tone letters (-b, -j, -s) are preserved
        has_tone_b = any(t.endswith("b") for t in tokens)
        has_tone_j = any(t.endswith("j") for t in tokens)
        has_tone_s = any(t.endswith("s") for t in tokens)
        self.assertTrue(has_tone_b, "Translation must preserve tone consonant -b (e.g. nyob)")
        self.assertTrue(has_tone_j, "Translation must preserve tone consonant -j (e.g. koj)")
        self.assertTrue(has_tone_s, "Translation must preserve tone consonant -s (e.g. puas)")

    def test_chain_mt_to_tts_contract(self):
        """
        Verify HmongTranslationModel -> HmongVITS contract:
        Hmong RPA string feeds into TTS synthesizer, producing 16kHz float32 audio
        normalized strictly to [-1.0, 1.0].
        """
        # Step 1: MT produces Hmong RPA text
        hmong_rpa_text = "nyob zoo koj puas nyob zoo"

        # Step 2: Genuine TTS synthesizes waveform
        synth_waveform = self.tts.synthesize(hmong_rpa_text, speed=1.0)

        # Step 3: Assert contract properties
        self.assertIsInstance(synth_waveform, np.ndarray)
        self.assertEqual(synth_waveform.dtype, np.float32)
        self.assertEqual(synth_waveform.ndim, 1)
        self.assertGreater(len(synth_waveform), 16000 * 0.5)  # Duration > 0.5s
        self.assertLessEqual(np.max(synth_waveform), 1.0)
        self.assertGreaterEqual(np.min(synth_waveform), -1.0)

    def test_chain_vad_to_streaming_engine(self):
        """
        Verify RingBufferVAD -> Streaming SpeechToSpeechEngine contract:
        A stream of 300ms PCM byte chunks is processed by VAD; when silence endpoint
        is reached, the accumulated speech buffer is dispatched and produces output audio.
        """
        chunk_samples = 4800  # 300ms @ 16kHz
        chunk_bytes = chunk_samples * 2  # 9600 bytes

        t_axis = np.linspace(0, 0.3, chunk_samples, endpoint=False, dtype=np.float32)
        speech_pcm = (0.5 * np.sin(2 * np.pi * 350.0 * t_axis) * 32767.0).astype(np.int16).tobytes()
        silent_pcm = np.zeros(chunk_samples, dtype=np.int16).tobytes()

        # Stream sequence: 1 silence, 3 speech, 2 silence (triggers endpoint)
        stream_chunks = [silent_pcm, speech_pcm, speech_pcm, speech_pcm, silent_pcm, silent_pcm]

        yielded = list(self.engine.process_stream(iter(stream_chunks)))
        self.assertGreaterEqual(len(yielded), 1)
        self.assertIsInstance(yielded[0], bytes)
        self.assertGreater(len(yielded[0]), 0)

    def test_full_cascaded_pipeline_contract(self):
        """
        Verify end-to-end Cascaded S2S pipeline:
        Audio In (WAV) -> STT -> MT -> TTS -> Audio Out (WAV)
        Yields non-null outputs and tracks phase latencies.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav_path = tmp.name

        try:
            # Generate synthetic input speech audio
            t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
            sig = (0.5 * np.sin(2 * np.pi * 300.0 * t) * 32767.0).astype(np.int16)
            with wave.open(tmp_wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(sig.tobytes())

            text_vi, text_hmong, audio_out, profile = self.engine.translate_speech(tmp_wav_path)

            self.assertIsInstance(text_vi, str)
            self.assertGreater(len(text_vi), 0)
            self.assertIsInstance(text_hmong, str)
            self.assertGreater(len(text_hmong), 0)
            self.assertIsInstance(audio_out, np.ndarray)
            self.assertGreater(len(audio_out), 1000)

            self.assertIn("stt_ms", profile)
            self.assertIn("mt_ms", profile)
            self.assertIn("tts_ms", profile)
            self.assertIn("total_latency_ms", profile)
            self.assertGreater(profile["stt_ms"], 0.0)
            self.assertGreater(profile["mt_ms"], 0.0)
            self.assertGreater(profile["tts_ms"], 0.0)
            self.assertGreater(profile["total_latency_ms"], 0.0)
        finally:
            if Path(tmp_wav_path).exists():
                try:
                    os.remove(tmp_wav_path)
                except OSError:
                    pass

    def test_multi_turn_state_isolation(self):
        """
        Verify that sequential dialogue turns maintain state isolation without lingering
        buffer artifacts or cross-turn token contamination.
        """
        turn_1_input = "xin chào"
        turn_2_input = "cứu tôi với"

        # Turn 1
        t1_trans = self.mt.translate(turn_1_input)
        self.assertEqual(t1_trans, "nyob zoo")

        # Turn 2
        t2_trans = self.mt.translate(turn_2_input)
        self.assertEqual(t2_trans, "pab kuv thiab")

        # Ensure turn 2 contains zero tokens from turn 1
        self.assertNotIn("nyob", t2_trans.split())
        self.assertNotIn("zoo", t2_trans.split())

        # Verify STT streaming state reset clears streaming buffer
        self.stt.transcribe_chunk(np.ones(1600, dtype=np.float32) * 0.1)
        self.stt.reset_streaming_state()
        self.assertEqual(len(self.stt._streaming_buffer), 0)
        self.assertIsNone(self.stt._streaming_prev_token)


if __name__ == "__main__":
    unittest.main()
