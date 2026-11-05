"""
tests/test_real_speech_stt.py
Unit and Integration Test Suite for Production Vietnamese Speech-to-Text (VietnameseSTT)
exercised on Authentic Vietnamese Speech Audio Datasets.

Tests:
  1. transcribe_file on authentic WAV audio files in tests/data/
     - sample1_greeting.wav: "xin chào bạn hôm nay bạn có khỏe không"
     - sample2_hospital.wav: "bác sĩ cho tôi hỏi bệnh viện ở đâu"
     - sample3_pharmacy.wav: "tôi cần mua thuốc cảm cúm và hạ sốt"
     - sample4_weather.wav: "hôm nay thời tiết rất đẹp và nhiều mây"
     - sample5_thanks.wav: "cảm ơn bạn rất nhiều hẹn gặp lại"
  2. Transcription accuracy, Vietnamese diacritics integrity, and non-empty guarantees.
  3. Format fidelity: WAV vs. compressed MP3 container transcription equivalence.
  4. Real-time streaming chunk transcription (transcribe_chunk) on audio buffers,
     state preservation, streaming reset, and silence handling.
  5. Multi-type input compatibility: file path, pathlib.Path, numpy array, torch.Tensor.
  6. Beam width variation (greedy beam_width=1 vs. prefix beam search beam_width=4).
  7. Robust boundary and error conditions (silence waveforms, non-existent files).

Conforms strictly to Project Master Specification PROJECT.md and Milestone P2-M4.
Zero stubs, zero mocks, 100% genuine model execution.
"""

import os
from pathlib import Path
import unittest
import numpy as np
import soundfile as sf
import torch

from src.models.stt_module import VietnameseSTT


class TestRealSpeechVietnameseSTT(unittest.TestCase):
    """
    Test suite verifying VietnameseSTT performance on realistic acoustic speech data.
    """

    @classmethod
    def setUpClass(cls):
        # Force CPU compute device to ensure deterministic execution and avoid GPU OOM in CI
        cls.device = "cpu"
        cls.stt = VietnameseSTT(device=cls.device)
        cls.data_dir = Path("tests/data")

        # Authoritative ground-truth transcripts for authentic sample files
        cls.expected_transcripts = {
            "sample1_greeting.wav": "xin chào bạn hôm nay bạn có khỏe không",
            "sample2_hospital.wav": "bác sĩ cho tôi hỏi bệnh viện ở đâu",
            "sample3_pharmacy.wav": "tôi cần mua thuốc cảm cúm và hạ sốt",
            "sample4_weather.wav": "hôm nay thời tiết rất đẹp và nhiều mây",
            "sample5_thanks.wav": "cảm ơn bạn rất nhiều hẹn gặp lại",
            "sample1_greeting.mp3": "xin chào bạn hôm nay bạn có khỏe không",
        }

        # Verify dataset files exist on disk
        for filename in cls.expected_transcripts.keys():
            file_path = cls.data_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Required audio sample file missing: {file_path}")

    def test_01_sample1_greeting_wav_transcription(self):
        """Verify transcribe_file accurately recognizes greeting utterance with correct diacritics."""
        file_path = self.data_dir / "sample1_greeting.wav"
        transcript = self.stt.transcribe_file(str(file_path))

        expected = self.expected_transcripts["sample1_greeting.wav"]
        self.assertIsInstance(transcript, str)
        self.assertGreater(len(transcript), 0, "Transcript should not be empty")
        self.assertEqual(transcript.strip(), expected)

        # Diacritics integrity check
        for char in ["à", "ạ", "ô", "ó", "ỏ"]:
            self.assertIn(char, transcript, f"Missing expected Vietnamese diacritic '{char}'")

    def test_02_sample2_hospital_wav_transcription(self):
        """Verify transcribe_file accurately recognizes clinical/hospital utterance."""
        file_path = self.data_dir / "sample2_hospital.wav"
        transcript = self.stt.transcribe_file(str(file_path))

        expected = self.expected_transcripts["sample2_hospital.wav"]
        self.assertIsInstance(transcript, str)
        self.assertGreater(len(transcript), 0, "Transcript should not be empty")
        self.assertEqual(transcript.strip(), expected)

        # Diacritics integrity check
        for char in ["á", "ĩ", "ỏ", "ệ", "ở", "đ", "â"]:
            self.assertIn(char, transcript, f"Missing expected Vietnamese diacritic '{char}'")

    def test_03_sample3_pharmacy_wav_transcription(self):
        """Verify transcribe_file accurately recognizes pharmacy/medication request utterance."""
        file_path = self.data_dir / "sample3_pharmacy.wav"
        transcript = self.stt.transcribe_file(str(file_path))

        expected = self.expected_transcripts["sample3_pharmacy.wav"]
        self.assertIsInstance(transcript, str)
        self.assertGreater(len(transcript), 0, "Transcript should not be empty")
        self.assertEqual(transcript.strip(), expected)

        # Diacritics integrity check
        for char in ["ô", "ầ", "ố", "ả", "ú", "à", "ạ"]:
            self.assertIn(char, transcript, f"Missing expected Vietnamese diacritic '{char}'")

    def test_04_sample4_weather_wav_transcription(self):
        """Verify transcribe_file accurately recognizes daily weather utterance."""
        file_path = self.data_dir / "sample4_weather.wav"
        transcript = self.stt.transcribe_file(str(file_path))

        expected = self.expected_transcripts["sample4_weather.wav"]
        self.assertIsInstance(transcript, str)
        self.assertGreater(len(transcript), 0, "Transcript should not be empty")
        self.assertEqual(transcript.strip(), expected)

        # Diacritics integrity check
        for char in ["ô", "ờ", "ế", "đ", "ẹ", "ề", "â"]:
            self.assertIn(char, transcript, f"Missing expected Vietnamese diacritic '{char}'")

    def test_05_sample5_thanks_wav_transcription(self):
        """Verify transcribe_file accurately recognizes gratitude utterance."""
        file_path = self.data_dir / "sample5_thanks.wav"
        transcript = self.stt.transcribe_file(str(file_path))

        expected = self.expected_transcripts["sample5_thanks.wav"]
        self.assertIsInstance(transcript, str)
        self.assertGreater(len(transcript), 0, "Transcript should not be empty")
        self.assertEqual(transcript.strip(), expected)

        # Diacritics integrity check
        for char in ["ả", "ơ", "ạ", "ề", "ẹ", "ặ"]:
            self.assertIn(char, transcript, f"Missing expected Vietnamese diacritic '{char}'")

    def test_06_mp3_vs_wav_format_equivalence(self):
        """Verify that MP3 compressed audio transcribes with identical accuracy to uncompressed WAV."""
        wav_path = self.data_dir / "sample1_greeting.wav"
        mp3_path = self.data_dir / "sample1_greeting.mp3"

        transcript_wav = self.stt.transcribe_file(str(wav_path))
        transcript_mp3 = self.stt.transcribe_file(str(mp3_path))

        self.assertEqual(
            transcript_mp3,
            transcript_wav,
            f"MP3 transcript '{transcript_mp3}' did not match WAV transcript '{transcript_wav}'"
        )
        self.assertEqual(
            transcript_mp3,
            self.expected_transcripts["sample1_greeting.mp3"]
        )

    def test_07_transcribe_chunk_streaming_slices(self):
        """Verify transcribe_chunk handles streaming audio chunks with state reset."""
        wav_path = self.data_dir / "sample1_greeting.wav"
        data, sr = sf.read(str(wav_path), dtype="float32")
        self.assertEqual(sr, 16000)

        self.stt.reset_streaming_state()
        chunk_size = int(sr * 0.3)  # 300ms chunks

        interim_transcripts = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            text = self.stt.transcribe_chunk(chunk)
            if text:
                interim_transcripts.append(text)

        # Streaming should yield non-empty interim updates
        self.assertGreater(len(interim_transcripts), 0, "Streaming should produce non-empty updates")

        # Verify state reset clears the streaming buffer and tokens
        self.stt.reset_streaming_state()
        self.assertEqual(len(self.stt._streaming_buffer), 0)
        self.assertIsNone(self.stt._streaming_prev_token)

    def test_08_transcribe_chunk_complete_speech_array(self):
        """Verify transcribe_chunk on a complete voiced numpy array."""
        wav_path = self.data_dir / "sample5_thanks.wav"
        data, sr = sf.read(str(wav_path), dtype="float32")

        self.stt.reset_streaming_state()
        text = self.stt.transcribe_chunk(data)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        self.assertIn("cảm ơn", text)

    def test_09_transcribe_chunk_silence_returns_empty(self):
        """Verify transcribe_chunk returns empty string on low-energy silence arrays."""
        silence = np.zeros(4800, dtype=np.float32)  # 300ms silence
        self.stt.reset_streaming_state()
        res = self.stt.transcribe_chunk(silence)
        self.assertEqual(res, "", "Silence chunk must yield empty transcript")

    def test_10_transcribe_file_direct_numpy_array(self):
        """Verify transcribe_file accepts raw numpy float32 arrays directly."""
        wav_path = self.data_dir / "sample1_greeting.wav"
        data, sr = sf.read(str(wav_path), dtype="float32")

        transcript = self.stt.transcribe_file(data)
        self.assertEqual(transcript.strip(), self.expected_transcripts["sample1_greeting.wav"])

    def test_11_transcribe_file_direct_torch_tensor(self):
        """Verify transcribe_file accepts torch.Tensor waveforms directly."""
        wav_path = self.data_dir / "sample2_hospital.wav"
        data, sr = sf.read(str(wav_path), dtype="float32")
        tensor = torch.from_numpy(data)

        # Test both 1D and 2D tensor representations
        transcript_1d = self.stt.transcribe_file(tensor)
        transcript_2d = self.stt.transcribe_file(tensor.unsqueeze(0))

        expected = self.expected_transcripts["sample2_hospital.wav"]
        self.assertEqual(transcript_1d.strip(), expected)
        self.assertEqual(transcript_2d.strip(), expected)

    def test_12_transcribe_file_pathlib_path(self):
        """Verify transcribe_file accepts pathlib.Path objects."""
        path_obj = self.data_dir / "sample4_weather.wav"
        transcript = self.stt.transcribe_file(path_obj)
        self.assertEqual(transcript.strip(), self.expected_transcripts["sample4_weather.wav"])

    def test_13_transcribe_file_beam_width_variation(self):
        """Verify both greedy (beam_width=1) and beam search (beam_width=4) execute cleanly."""
        wav_path = self.data_dir / "sample5_thanks.wav"
        res_greedy = self.stt.transcribe_file(str(wav_path), beam_width=1)
        res_beam = self.stt.transcribe_file(str(wav_path), beam_width=4)

        expected = self.expected_transcripts["sample5_thanks.wav"]
        self.assertEqual(res_greedy.strip(), expected)
        self.assertEqual(res_beam.strip(), expected)

    def test_14_silence_audio_transcription(self):
        """Verify transcribe_file returns empty string for silence tensor without crashing."""
        silence = np.zeros(16000, dtype=np.float32)  # 1 second of pure silence
        transcript = self.stt.transcribe_file(silence)
        self.assertEqual(transcript, "")

    def test_15_nonexistent_audio_file_raises_error(self):
        """Verify transcribe_file raises appropriate exception when file does not exist."""
        with self.assertRaises(Exception):
            self.stt.transcribe_file("tests/data/nonexistent_audio_file_12345.wav")


if __name__ == "__main__":
    unittest.main()
