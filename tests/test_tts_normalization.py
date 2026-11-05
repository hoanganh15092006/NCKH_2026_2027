"""
tests/test_tts_normalization.py
Unit and Regression Test Suite for HmongVITS Audio Normalization and WAV Serialization.

Verifies:
  1. 2-Stage DC removal and target RMS normalization (-18.0 dBFS) across diverse phrases.
  2. Absolute bounds on synthesized float32 audio: -20.0 dBFS <= RMS <= -16.0 dBFS.
  3. Strict zero-centering: mean DC offset < 1e-3.
  4. Peak headroom protection: peak amplitude strictly <= 0.99.
  5. Standard 44-byte compliant RIFF/WAV header (1 channel, 16-bit PCM, 16000 Hz).
  6. Clean audio decoding via soundfile without errors or distortion.
"""

import io
import unittest
import wave
import numpy as np
import soundfile as sf
from src.models.tts_module import HmongVITS


class TestHmongVITSNormalization(unittest.TestCase):
    """Test suite for HmongVITS audio loudness normalization and WAV packaging."""

    @classmethod
    def setUpClass(cls):
        cls.tts = HmongVITS(sample_rate=16000, device="cpu")
        cls.test_phrases = [
            "nyob zoo",
            "ua tsaug",
            "kuv nyob zoo",
            "sib ntsib dua",
            "koj tuaj qhov twg tuaj",
            "hmoob dawb",
            "hnub no huab cua zoo heev",
        ]

    def test_01_rms_loudness_and_dc_bounds(self):
        """Verify synthesized float32 waveform achieves target RMS between -20 and -16 dBFS with 0 DC offset."""
        for phrase in self.test_phrases:
            audio = self.tts.synthesize(phrase)
            self.assertIsInstance(audio, np.ndarray)
            self.assertEqual(audio.dtype, np.float32)
            self.assertGreater(len(audio), 0, f"Synthesized audio for '{phrase}' was empty")

            dc = float(np.mean(audio))
            peak = float(np.max(np.abs(audio)))
            rms = float(np.sqrt(np.mean(audio ** 2)))
            dbfs = 20.0 * np.log10(rms) if rms > 0 else -999.0

            # 1. Audible loudness within standard web playback corridor [-20.0, -16.0] dBFS
            self.assertGreaterEqual(
                dbfs, -20.0,
                f"Audio for '{phrase}' too quiet: {dbfs:.2f} dBFS < -20.0 dBFS"
            )
            self.assertLessEqual(
                dbfs, -16.0,
                f"Audio for '{phrase}' too loud: {dbfs:.2f} dBFS > -16.0 dBFS"
            )

            # 2. DC offset eliminated
            self.assertLess(
                abs(dc), 1e-3,
                f"Audio for '{phrase}' has excessive DC offset: {dc:.6f}"
            )

            # 3. Peak headroom protection
            self.assertLessEqual(
                peak, 0.99,
                f"Audio for '{phrase}' exceeded safety ceiling: {peak:.4f} > 0.99"
            )

    def test_02_wav_header_and_pcm16_integrity(self):
        """Verify synthesize_to_bytes generates 100% compliant RIFF/WAV 16-bit 16kHz mono audio."""
        for phrase in self.test_phrases:
            wav_bytes = self.tts.synthesize_to_bytes(phrase, format="wav")
            self.assertIsInstance(wav_bytes, bytes)
            self.assertGreater(len(wav_bytes), 44, "WAV stream shorter than 44-byte header")

            # Check RIFF header via standard wave module
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1, "Must be mono channel")
                self.assertEqual(wf.getsampwidth(), 2, "Must be 16-bit (2 bytes/sample)")
                self.assertEqual(wf.getframerate(), 16000, "Must be 16000 Hz")
                n_frames = wf.getnframes()
                self.assertGreater(n_frames, 0, "WAV frames must be > 0")

            # Verify soundfile cleanly reads generated WAV bytes without warnings or errors
            data, sr = sf.read(io.BytesIO(wav_bytes))
            self.assertEqual(sr, 16000)
            self.assertEqual(len(data), n_frames)

            data_rms = float(np.sqrt(np.mean(data ** 2)))
            data_dbfs = 20.0 * np.log10(data_rms) if data_rms > 0 else -999.0
            data_peak = float(np.max(np.abs(data)))
            data_dc = float(np.mean(data))

            self.assertGreaterEqual(data_dbfs, -20.0)
            self.assertLessEqual(data_dbfs, -16.0)
            self.assertLess(abs(data_dc), 1e-3)
            self.assertLessEqual(data_peak, 0.99)

    def test_03_empty_and_whitespace_input(self):
        """Verify empty and whitespace inputs return empty audio without crashing."""
        for empty_text in ["", "   ", "\t\n"]:
            audio = self.tts.synthesize(empty_text)
            self.assertEqual(len(audio), 0)

            pcm_bytes = self.tts.synthesize_to_bytes(empty_text, format="pcm")
            self.assertEqual(len(pcm_bytes), 0)


if __name__ == "__main__":
    unittest.main()
