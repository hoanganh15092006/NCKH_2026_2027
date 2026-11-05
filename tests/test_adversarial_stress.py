"""
tests/test_adversarial_stress.py
Empirical Adversarial Stress Testing Suite for Vietnamese-to-Hmong Daw S2S System.

Target Components:
1. AudioPreprocessor:
   - Pure silence handling (zero amplitude, gain explosion guard, mel extraction, VAD bias anomaly)
   - Maximum clipping and overload (+1000.0 amplitude, clipping suppression < 0.99)
   - Arbitrary sample rates (8k, 44.1k, 48k, 96k, multi-channel stereo/surround downmix)
   - NaN / Inf tensor handling (empirical behavior on un-sanitized floating point values)
   - Random white noise (STE vs Silero VAD, filter_silence fallback behavior)

2. HmongTranslationModel & RPATextNormalizer:
   - Empty strings and whitespace variations
   - Pure numbers and numerical tokens
   - Strings missing tone markers (mid-level tone [33] phonotactic validation)
   - Excessive repetition inputs (repetition penalty theta=1.20, n-gram blocking, termination)
   - Long sentences > 500 characters (both < 512 tokens and > 512 tokens boundary crash)

3. HmongVITS:
   - Unusual phoneme combinations, illegal symbols, emojis, and vowel-less clusters
   - Rapid successive synthesis calls (latency stability and consistency)
   - Memory footprint stability across successive synthesis passes
"""

import math
import sys
import os
import time
import tracemalloc
import unittest
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torchaudio

from src.data.audio_preprocessor import AudioPreprocessor
from src.models.mt_module import HmongTranslationModel, RPATextNormalizer
from src.models.tts_module import HmongVITS, tokenize_hmong_rpa


class TestAudioPreprocessorAdversarialStress(unittest.TestCase):
    """Adversarial stress testing on AudioPreprocessor."""

    @classmethod
    def setUpClass(cls):
        cls.preprocessor = AudioPreprocessor()

    def test_audio_pure_silence_all_zeros(self):
        """Stress Test 1.1: AudioPreprocessor behavior on pure digital silence (zero amplitude)."""
        sr = 16000
        total_samples = sr * 2 # 32000 samples (62 full chunks of 512 + 256 remainder)
        silence = torch.zeros(total_samples)

        # 1. Test normalize_rms
        norm = self.preprocessor.normalize_rms(silence)
        self.assertEqual(norm.shape, silence.shape)
        self.assertFalse(torch.isnan(norm).any().item())
        self.assertFalse(torch.isinf(norm).any().item())
        self.assertEqual(norm.abs().max().item(), 0.0)

        # 2. Test extract_mel_spectrogram
        mel = self.preprocessor.extract_mel_spectrogram(silence)
        self.assertEqual(mel.dim(), 3)
        self.assertEqual(mel.size(1), 80)
        self.assertFalse(torch.isnan(mel).any().item())
        self.assertFalse(torch.isinf(mel).any().item())
        # All values should be clamped to log(1e-5) ~= -11.5129
        self.assertAlmostEqual(mel.min().item(), -11.5129, delta=0.01)
        self.assertAlmostEqual(mel.max().item(), -11.5129, delta=0.01)

        # 3. Test STE VAD (Energy-based VAD correctly marks silence as False)
        ste_vad = self.preprocessor.detect_voice_activity_ste(silence)
        self.assertFalse(ste_vad.any().item())

        # 4. Silence filter behavior observation:
        # EMPIRICAL FINDING: SileroVADWrapper is instantiated with unseeded random weights.
        # Its forward pass on digital silence produces a probability hovering around 0.50 (e.g., 0.48 - 0.53).
        # When prob >= 0.50, silence is treated as speech (sliced to 31744 samples).
        # When prob < 0.50, fallback triggers (retaining 32000 samples).
        filt, intervals = self.preprocessor.filter_silence(silence)
        self.assertEqual(intervals[0][0], 0)
        self.assertIn(intervals[0][1], (31744, 32000))
        self.assertEqual(filt.shape, (1, intervals[0][1]))

    def test_audio_maximum_clipping_and_saturation(self):
        """Stress Test 1.2: Extreme amplitude overload (+1000.0) and clipping prevention."""
        sr = 16000
        # Extreme square wave overload
        clipped_signal = torch.sign(torch.randn(sr)) * 1000.0

        norm = self.preprocessor.normalize_rms(clipped_signal, target_db=-20.0)
        max_val = norm.abs().max().item()
        self.assertLessEqual(max_val, 0.990001)
        self.assertFalse(torch.isnan(norm).any().item())
        self.assertFalse(torch.isinf(norm).any().item())

        # Mel extraction on heavily clipped audio
        mel = self.preprocessor.extract_mel_spectrogram(norm)
        self.assertFalse(torch.isnan(mel).any().item())
        self.assertFalse(torch.isinf(mel).any().item())

    def test_audio_arbitrary_sample_rates_and_channels(self):
        """Stress Test 1.3: Arbitrary sample rates (8k, 44.1k, 48k, 96k) and multi-channel downmix."""
        sample_rates = [8000, 11025, 22050, 44100, 48000, 96000]
        for sr in sample_rates:
            duration_s = 0.5
            n_samples = int(sr * duration_s)
            t = torch.linspace(0, duration_s, n_samples)
            # 6-channel audio (e.g. 5.1 surround)
            multi_channel_wav = torch.stack([
                torch.sin(2 * math.pi * (100 * (i + 1)) * t)
                for i in range(6)
            ], dim=0)

            resampled = self.preprocessor.resample_to_16k(multi_channel_wav, orig_sr=sr)
            self.assertEqual(resampled.dim(), 2)
            self.assertEqual(resampled.size(0), 1) # Mono channel
            expected_samples = int(16000 * duration_s)
            self.assertAlmostEqual(resampled.size(1), expected_samples, delta=100)
            self.assertFalse(torch.isnan(resampled).any().item())

    def test_audio_nan_and_inf_tensor_propagation(self):
        """Stress Test 1.4: Empirical behavior when fed NaN and Inf values."""
        sr = 16000
        nan_tensor = torch.tensor([float("nan")] * sr)
        inf_tensor = torch.tensor([float("inf")] * sr)

        # Observation: normalize_rms propagates NaN without raising an exception
        norm_nan = self.preprocessor.normalize_rms(nan_tensor)
        self.assertTrue(torch.isnan(norm_nan).all().item())

        norm_inf = self.preprocessor.normalize_rms(inf_tensor)
        self.assertTrue(torch.isnan(norm_inf).all().item())

        # Mel spectrogram on NaN yields full NaN tensor
        mel_nan = self.preprocessor.extract_mel_spectrogram(nan_tensor)
        self.assertEqual(torch.isnan(mel_nan).sum().item(), mel_nan.numel())

        # Silero VAD on NaN produces all False because comparisons against NaN evaluate to False
        vad_nan = self.preprocessor.detect_voice_activity_silero(nan_tensor)
        self.assertFalse(vad_nan.any().item())

    def test_audio_random_white_noise(self):
        """Stress Test 1.5: Random Gaussian white noise across extreme amplitudes."""
        sr = 16000
        noise_levels = [0.001, 0.05, 0.5, 2.0, 10.0]
        for amp in noise_levels:
            noise = torch.randn(sr) * amp
            norm_noise = self.preprocessor.normalize_rms(noise, target_db=-20.0)
            self.assertLessEqual(norm_noise.abs().max().item(), 0.990001)

            # Check STE VAD on uniform noise
            ste_vad = self.preprocessor.detect_voice_activity_ste(noise)
            self.assertIsInstance(ste_vad, torch.Tensor)

            # Mel spectrogram extraction under noise
            mel_noise = self.preprocessor.extract_mel_spectrogram(norm_noise)
            self.assertEqual(mel_noise.size(1), 80)
            self.assertFalse(torch.isnan(mel_noise).any().item())


class TestHmongTranslationAndNormalizerAdversarialStress(unittest.TestCase):
    """Adversarial stress testing on HmongTranslationModel and RPATextNormalizer."""

    @classmethod
    def setUpClass(cls):
        cls.normalizer = RPATextNormalizer()
        cls.mt = HmongTranslationModel()

    def test_mt_and_normalizer_empty_and_whitespace_strings(self):
        """Stress Test 2.1: Empty strings, single whitespace, and multiline control characters."""
        empty_inputs = ["", "   ", "\t", "\n", "\r\n", "   \t\n  \r  "]
        for inp in empty_inputs:
            norm_vi = self.normalizer.normalize_vietnamese(inp)
            self.assertEqual(norm_vi, "")

            norm_hmong = self.normalizer.normalize_hmong_rpa(inp)
            self.assertEqual(norm_hmong, "")

            tokens = self.normalizer.validate_rpa_tokens(inp)
            self.assertEqual(tokens, [])

            trans = self.mt.translate(inp)
            self.assertEqual(trans, "")

        batch_res = self.mt.translate_batch(empty_inputs)
        self.assertEqual(batch_res, [""] * len(empty_inputs))

    def test_mt_and_normalizer_pure_numbers_and_punctuation(self):
        """Stress Test 2.2: Pure numbers, floating point numbers, and mathematical symbols."""
        num_inputs = [
            "12345",
            "12345 67890",
            "3.14159265",
            "2026-10-19",
            "100% 50$ 200€",
            "99999999999999999999"
        ]
        for num_str in num_inputs:
            norm_vi = self.normalizer.normalize_vietnamese(num_str)
            self.assertIsInstance(norm_vi, str)
            self.assertTrue(len(norm_vi) > 0)

            # Translation handles numbers gracefully without unhandled exception
            trans = self.mt.translate(num_str)
            self.assertIsInstance(trans, str)
            self.assertTrue(len(trans) > 0)

    def test_mt_and_normalizer_strings_missing_tone_markers(self):
        """Stress Test 2.3: Hmong words missing tone markers (mid level [33] phonotactic rules)."""
        # In Hmong Daw RPA, syllables without tone consonants (-b, -m, -d, -j, -v, -s, -g) represent mid tone [33]
        mid_tone_phrases = [
            "no",            # this (mid tone)
            "zoo",           # good (mid tone)
            "lub tsev no zoo heev", # mixed mid tone and marked tone
            "kuv mus tsev no"
        ]
        for phrase in mid_tone_phrases:
            validated = self.normalizer.validate_rpa_tokens(phrase)
            self.assertTrue(len(validated) > 0)
            for token_info in validated:
                clean = token_info["clean"]
                if clean in ("no", "zoo"):
                    self.assertEqual(token_info["tone"], "")
                    self.assertIn("33 mid level", token_info["tone_desc"])
                self.assertTrue(token_info["is_rpa"])

    def test_mt_excessive_repetition_inputs(self):
        """Stress Test 2.4: Excessive repeated phrases verifying termination and bounded beam generation."""
        repeated_input = "xin chào bạn " * 30
        trans = self.mt.translate(repeated_input, repetition_penalty=1.20, no_repeat_ngram_size=3)
        self.assertIsInstance(trans, str)
        self.assertTrue(len(trans) > 0)

        # Verify that translation completes without infinite loop and length is bounded by max_length
        tokens = trans.split()
        self.assertLessEqual(len(tokens), 48)

    def test_mt_long_sentences_under_and_over_pos_limit(self):
        """Stress Test 2.5: Long sentences > 500 characters and > 512 tokens boundary crash test."""
        # 1. Long sentence > 500 characters but < 512 tokens (SUCCEEDS)
        long_sentence_sub_limit = (
            "hôm nay tôi đi đến bệnh viện gặp bác sĩ để khám bệnh và uống thuốc "
            "vì tôi bị đau đầu đau bụng đau ngực rất nhiều lần và cần sự giúp đỡ "
            "của bác sĩ tại bệnh viện này để chữa trị sức khỏe cho tôi và gia đình tôi "
        ) * 3
        self.assertGreater(len(long_sentence_sub_limit), 500)
        self.assertLess(len(long_sentence_sub_limit.split()), 500)

        trans_long = self.mt.translate(long_sentence_sub_limit)
        self.assertIsInstance(trans_long, str)
        self.assertTrue(len(trans_long) > 0)

        # 2. Sentence exceeding the 512-token positional embedding matrix (CONFIRMED FAILURE MODE)
        oversized_sentence = "hôm nay tôi đi " * 150 # 600 words + 2 special tokens = 602 tokens > 512
        self.assertGreater(len(oversized_sentence.split()), 512)

        # Empirical challenge check: verifies positional embedding mismatch RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            self.mt.translate(oversized_sentence)
        self.assertIn("must match the size of tensor b (512)", str(ctx.exception))


class TestHmongVITSSynthesisAdversarialStress(unittest.TestCase):
    """Adversarial stress testing on HmongVITS synthesizer."""

    @classmethod
    def setUpClass(cls):
        cls.vits = HmongVITS()

    def test_vits_unusual_phonemes_and_illegal_characters(self):
        """Stress Test 3.1: Unusual phonemes, non-RPA characters, alien symbols, and emojis."""
        adversarial_inputs = [
            "!!!???---...",
            "#@!$%^&*()",
            "12345 67890",
            "strng plh ntxh",
            "xyz abcdef ghijkl",
            "nyob zoo koj hôm nay bạn khỏe không",
            "a" * 100,
            "bbbbbbbbbb",
            "kuv123mob456plab789",
            "ñçáéíóú 測試 🚀🎉"
        ]

        for text in adversarial_inputs:
            tokens = tokenize_hmong_rpa(text)
            self.assertIsInstance(tokens, list)

            audio = self.vits.synthesize(text)
            self.assertIsInstance(audio, np.ndarray)
            self.assertEqual(audio.dtype, np.float32)

            if len(audio) > 0:
                self.assertFalse(np.isnan(audio).any())
                self.assertFalse(np.isinf(audio).any())
                self.assertGreaterEqual(audio.min(), -1.0)
                self.assertLessEqual(audio.max(), 1.0)

    def test_vits_rapid_successive_calls_latency_stability(self):
        """Stress Test 3.2: 50 rapid sequential synthesis calls measuring latency stability."""
        num_calls = 50
        latencies = []
        test_phrases = [
            "nyob zoo koj puas nyob zoo",
            "kuv yog neeg hmoob dawb",
            "pab kuv thiab kuv mob plab",
            "tsev kho mob kws kho mob"
        ]

        for i in range(num_calls):
            phrase = test_phrases[i % len(test_phrases)]
            t0 = time.perf_counter()
            audio = self.vits.synthesize(phrase)
            latencies.append(time.perf_counter() - t0)

            self.assertGreater(len(audio), 0)
            self.assertFalse(np.isnan(audio).any())

        mean_ms = np.mean(latencies) * 1000.0
        p90_ms = np.percentile(latencies, 90) * 1000.0

        # High throughput verification (< 150ms per phrase on CPU)
        self.assertLess(mean_ms, 150.0)
        self.assertLess(p90_ms, 200.0)

    def test_vits_memory_stability_tracemalloc(self):
        """Stress Test 3.3: Memory stability across repeated synthesis passes."""
        tracemalloc.start()
        snapshot_start = tracemalloc.take_snapshot()

        for _ in range(40):
            _ = self.vits.synthesize("nyob zoo koj puas nyob zoo kuv yog neeg hmoob")

        snapshot_end = tracemalloc.take_snapshot()
        stats = snapshot_end.compare_to(snapshot_start, "filename")

        # Sum allocated differences across all files
        total_growth_kb = sum(s.size_diff for s in stats if s.size_diff > 0) / 1024.0

        tracemalloc.stop()

        # Ensure memory growth over 40 synthesis calls is bounded (< 25MB on CPU)
        self.assertLess(total_growth_kb, 25600.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
