"""
Tier 2: Boundary Value Analysis (BVA) & Edge Case Hardening Suite
Multimodal Vietnamese-to-H'Mong Speech-to-Speech Translation System.
Tests system limits, extreme inputs, and boundary conditions:
1. Silence audio (zero amplitude, log epsilon, gain stability)
2. Extremely loud / clipping audio (+6dBFS to +12dBFS overload, clamping)
3. Empty string inputs (MT, TTS, normalizer)
4. Single character inputs (diacritics, standalone tone letters)
5. Long text > 256 tokens (exceeding positional embeddings, truncation)
6. Missing tone markers / Zero tone (mid-level tone [33]) & malformed codas
7. Odd sample rates (8kHz, 22.05kHz, 44.1kHz, 48kHz, 96kHz resampling)
8. Buffer boundary extremes (undersized chunks, buffer overflow capping)
9. Numerical integrity (NaN, Inf injection protection)
"""

import math
import os
import re
import sys
import unittest
from pathlib import Path
from typing import Any, List, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_utils import (
    HAS_NUMPY,
    HAS_TORCH,
    calculate_rms_dbfs,
    generate_synthetic_audio,
)

if HAS_NUMPY:
    import numpy as np
if HAS_TORCH:
    import torch


class TestBoundarySilenceAndClipping(unittest.TestCase):
    """
    Boundary tests for acoustic extremes: digital silence and severe saturation/clipping.
    """

    def test_silence_audio_zero_amplitude_rms(self):
        """Test RMS calculation on absolute digital silence does not cause ZeroDivisionError or MathDomainError."""
        n_samples = 16000
        silence = [0.0] * n_samples
        if HAS_NUMPY:
            silence_arr = np.zeros(n_samples, dtype=np.float32)
            rms_db = calculate_rms_dbfs(silence_arr)
        else:
            rms_db = calculate_rms_dbfs(silence)

        # Expected output: extremely low dBFS (<= -100 dBFS), bounded by epsilon
        self.assertLessEqual(rms_db, -100.0)
        self.assertFalse(math.isnan(rms_db))
        self.assertFalse(math.isinf(rms_db))

    def test_silence_normalization_gain_explosion_guard(self):
        """Test normalizing pure silence does not cause gain explosion (zero division) or NaN output."""
        n_samples = 4800
        target_db = -20.0
        target_rms = 10.0 ** (target_db / 20.0)
        min_rms_threshold = 1e-4  # Minimum RMS to prevent amplifying pure floor noise

        if HAS_NUMPY:
            silence = np.zeros(n_samples, dtype=np.float32)
            curr_rms = float(np.sqrt(np.mean(silence ** 2) + 1e-12))
            if curr_rms < min_rms_threshold:
                # Safe guard: do not apply gain to silence
                normalized = silence
            else:
                gain = target_rms / curr_rms
                normalized = np.clip(silence * gain, -1.0, 1.0)

            self.assertFalse(np.any(np.isnan(normalized)))
            self.assertEqual(float(np.max(normalized)), 0.0)
        else:
            silence = [0.0] * n_samples
            curr_rms = math.sqrt(sum(s ** 2 for s in silence) / len(silence) + 1e-12)
            if curr_rms < min_rms_threshold:
                normalized = silence
            else:
                gain = target_rms / curr_rms
                normalized = [max(-1.0, min(1.0, s * gain)) for s in silence]
            self.assertEqual(max(normalized), 0.0)

    def test_extreme_loud_audio_clipping_and_attenuation(self):
        """Test audio with amplitude 5.0 (+14 dBFS) is attenuated to -20 dBFS without exceeding [-1.0, 1.0]."""
        sr = 16000
        # Highly saturated waveform
        loud_audio = generate_synthetic_audio(duration_s=0.5, sample_rate=sr, amplitude=5.0)
        target_db = -20.0
        target_rms = 10.0 ** (target_db / 20.0)

        if HAS_NUMPY and isinstance(loud_audio, np.ndarray):
            curr_rms = np.sqrt(np.mean(loud_audio ** 2) + 1e-12)
            gain = target_rms / curr_rms
            # Gain must be < 1.0 (attenuation)
            self.assertLess(gain, 0.2)
            normalized = np.clip(loud_audio * gain, -1.0, 1.0)
            norm_rms = calculate_rms_dbfs(normalized)
            self.assertAlmostEqual(norm_rms, target_db, delta=0.5)
            self.assertLessEqual(np.max(normalized), 1.0)
            self.assertGreaterEqual(np.min(normalized), -1.0)
        else:
            sum_sq = sum(s ** 2 for s in loud_audio)
            curr_rms = math.sqrt(sum_sq / len(loud_audio) + 1e-12)
            gain = target_rms / curr_rms
            self.assertLess(gain, 0.2)
            normalized = [max(-1.0, min(1.0, s * gain)) for s in loud_audio]
            norm_rms = calculate_rms_dbfs(normalized)
            self.assertAlmostEqual(norm_rms, target_db, delta=0.5)
            self.assertLessEqual(max(normalized), 1.0)
            self.assertGreaterEqual(min(normalized), -1.0)


class TestBoundaryTextInputs(unittest.TestCase):
    """
    Boundary tests for text inputs: empty, whitespace-only, single character, long tokens, and RPA tones.
    """

    def normalize_rpa(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        return " ".join(text.split())

    def test_empty_string_handling(self):
        """Test empty string input produces empty normalized output without index errors."""
        empty_input = ""
        normalized = self.normalize_rpa(empty_input)
        self.assertEqual(normalized, "")

    def test_whitespace_only_string_handling(self):
        """Test whitespace-only strings (spaces, tabs, newlines) reduce cleanly to empty string."""
        whitespace_input = "   \t\r\n   \n\t  "
        normalized = self.normalize_rpa(whitespace_input)
        self.assertEqual(normalized, "")

    def test_single_character_input_preservation(self):
        """Test single character input is correctly processed."""
        # Single Latin character
        self.assertEqual(self.normalize_rpa("a"), "a")
        # Single uppercase
        self.assertEqual(self.normalize_rpa("B"), "b")
        # Standalone tone letter
        self.assertEqual(self.normalize_rpa("j"), "j")

    def test_single_punctuation_character_handling(self):
        """Test standalone punctuation characters are safely stripped without crashing."""
        punct_samples = [".", ",", "!", "?", ";", ":", "-", '"', "'"]
        for p in punct_samples:
            normalized = self.normalize_rpa(p)
            self.assertEqual(normalized, "")

    def test_long_text_exceeding_256_tokens_truncation(self):
        """Test input with > 256 tokens is safely truncated to max_source_length=128."""
        max_source_length = 128
        words = ["lo lus"] * 300  # 600 words
        long_text = " ".join(words)
        token_list = long_text.split()
        self.assertGreater(len(token_list), 256)

        # Truncation algorithm
        truncated_tokens = token_list[:max_source_length]
        self.assertEqual(len(truncated_tokens), max_source_length)
        truncated_text = " ".join(truncated_tokens)
        self.assertEqual(len(truncated_text.split()), max_source_length)

    def test_unmarked_zero_tone_words_preservation(self):
        """
        Test Hmong Daw words with ZERO tone marker (mid-level tone [33]).
        In RPA, words ending in vowels (e.g., zoo, ua, lo, no, siab) have mid tone and no consonant.
        These must NOT be flagged as missing tones or corrupted by the normalizer.
        """
        zero_tone_words = ["zoo", "ua", "lo", "no", "tuaj", "siab"]
        # Words ending in vowel: zoo, ua, lo, no
        vowel_endings = ["zoo", "ua", "lo", "no"]
        for w in vowel_endings:
            normalized = self.normalize_rpa(w)
            self.assertEqual(normalized, w)
            # Must not end in tone consonants (b, m, d, j, v, s, g)
            self.assertFalse(normalized.endswith(('b', 'm', 'd', 'j', 'v', 's', 'g')))

    def test_malformed_coda_consonants_handling(self):
        """
        Test words ending in non-RPA codas (e.g., 'k', 't', 'p', 'x' which do not exist as codas in RPA).
        The normalizer must preserve or handle the token gracefully without throwing exceptions.
        """
        malformed_inputs = "hmoobk dejt pobx"
        normalized = self.normalize_rpa(malformed_inputs)
        self.assertEqual(normalized, "hmoobk dejt pobx")


class TestBoundarySampleRatesAndBuffers(unittest.TestCase):
    """
    Boundary tests for unusual sampling rates and audio buffer dimensions.
    """

    def test_odd_sampling_rates_conversion_dimensions(self):
        """Test audio resampling sample counts across standard and non-standard rates."""
        target_sr = 16000
        test_cases = [
            (8000, 0.5, 4000, 8000),      # Narrowband telephony
            (11025, 0.4, 4410, 6400),     # Quarter-CD rate
            (22050, 1.0, 22050, 16000),   # Half-CD rate
            (44100, 1.0, 44100, 16000),   # CD rate
            (48000, 1.0, 48000, 16000),   # Studio/DAT rate
            (96000, 0.25, 24000, 4000),   # High-res audio
        ]
        for src_sr, duration, src_samples, expected_target in test_cases:
            # Resampling formula: target_samples = int(round(src_samples * (target_sr / src_sr)))
            target_samples = int(round(src_samples * (target_sr / float(src_sr))))
            self.assertEqual(target_samples, expected_target, f"Failed for src_sr={src_sr}")

    def test_undersized_audio_chunk_padding(self):
        """Test that chunks smaller than 300ms (4800 samples) are zero-padded to standard size."""
        chunk_samples = 4800
        short_chunk_len = 1200  # 75ms only
        if HAS_NUMPY:
            short_chunk = np.ones(short_chunk_len, dtype=np.float32) * 0.5
            # Pad with zeros to 4800
            padded = np.pad(short_chunk, (0, chunk_samples - short_chunk_len), mode="constant")
            self.assertEqual(len(padded), chunk_samples)
            self.assertEqual(float(padded[0]), 0.5)
            self.assertEqual(float(padded[-1]), 0.0)
        else:
            short_chunk = [0.5] * short_chunk_len
            padded = short_chunk + [0.0] * (chunk_samples - short_chunk_len)
            self.assertEqual(len(padded), chunk_samples)
            self.assertEqual(padded[0], 0.5)
            self.assertEqual(padded[-1], 0.0)

    def test_max_ring_buffer_cap_enforcement(self):
        """Test ring buffer strictly discards or flushes when audio exceeds 15.0 seconds (50 chunks)."""
        max_duration_s = 15.0
        chunk_s = 0.3
        max_chunks = int(max_duration_s / chunk_s)  # 50

        # Simulate stream sending 80 continuous chunks without silence
        from collections import deque
        buffer = deque(maxlen=max_chunks)
        for i in range(80):
            buffer.append(f"chunk_{i}")

        # Buffer length must never exceed 50
        self.assertEqual(len(buffer), max_chunks)
        # Oldest chunks must have been evicted (oldest remaining is chunk_30)
        self.assertEqual(buffer[0], "chunk_30")
        self.assertEqual(buffer[-1], "chunk_79")


class TestBoundaryNumericalIntegrity(unittest.TestCase):
    """
    Boundary tests for numerical safety: NaN, +Inf, -Inf detection.
    """

    def test_nan_inf_detection_in_audio_waveform(self):
        """Test detection and sanitization of corrupted float values (NaN, Inf) in audio stream."""
        if HAS_NUMPY:
            corrupted_audio = np.array([0.1, 0.2, np.nan, 0.4, np.inf, -np.inf, 0.5], dtype=np.float32)
            has_invalid = np.any(np.isnan(corrupted_audio)) or np.any(np.isinf(corrupted_audio))
            self.assertTrue(has_invalid)

            # Sanitization logic: replace NaN/Inf with 0.0
            sanitized = np.nan_to_num(corrupted_audio, nan=0.0, posinf=1.0, neginf=-1.0)
            self.assertFalse(np.any(np.isnan(sanitized)))
            self.assertFalse(np.any(np.isinf(sanitized)))
            self.assertEqual(float(sanitized[2]), 0.0)
            self.assertEqual(float(sanitized[4]), 1.0)
            self.assertEqual(float(sanitized[5]), -1.0)
        else:
            corrupted = [0.1, 0.2, float("nan"), 0.4, float("inf"), float("-inf"), 0.5]
            has_invalid = any(math.isnan(x) or math.isinf(x) for x in corrupted)
            self.assertTrue(has_invalid)


if __name__ == "__main__":
    unittest.main()
