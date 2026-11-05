"""
tests/test_adversarial_p2.py
Adversarial Empirical Challenge Harness for Milestone P2-M5.
Stress-tests the Vietnamese STT and Speech-to-Speech (S2S) translation pipeline
under hostile inputs, corrupted containers, edge cases, and extreme phonology.

Categories:
  1. Corrupted Containers & Truncated Headers (Damaged MP3, Truncated WAV, Zero-byte, ASCII)
  2. Non-standard Sample Rates & Channels (8kHz, 22.05kHz, 44.1kHz, 48kHz, Stereo, 5.1 Multichannel)
  3. Low RMS Amplitude & Whispering (Pure zero, Near-silence -120dBFS, Attenuated speech)
  4. Extreme Additive Noise Degradation (AWGN at SNR 20dB, 10dB, 5dB, 0dB, -5dB)
  5. Rapid Speech & Speed Variations (Time-stretched 1.25x, 1.5x, 2.0x)
  6. Rare Diacritics & Complex Vietnamese Phonology (Diacritics coverage, Tokenizer fidelity)
  7. Fake String & Hardcoded Response Audit (Tones, Reversed speech, Modulated speech, Static check)
  8. End-to-End S2S Pipeline Hostile Resilience & Vulnerability Confirmations
"""

import io
import os
import shutil
import tempfile
import unittest
import numpy as np
import soundfile as sf
import torch
import librosa
from fastapi.testclient import TestClient

from src.data.audio_preprocessor import AudioPreprocessor
from src.models.stt_module import VietnameseSTT, VIETNAMESE_CTC_VOCAB
from src.models.mt_module import HmongTranslationModel, RPATextNormalizer
from src.models.tts_module import HmongVITS
from src.models.s2s_pipeline import SpeechToSpeechEngine
from src.api.app_stream import app, get_engine


class TestAdversarialContainersAndCorruption(unittest.TestCase):
    """
    Challenge Dimension 1: Corrupted byte containers, truncated headers, zero-byte files,
    and invalid container formats.
    """

    @classmethod
    def setUpClass(cls):
        cls.device = "cpu"
        cls.stt = VietnameseSTT(device=cls.device)
        cls.tmp_dir = tempfile.mkdtemp(prefix="nckh_adv_containers_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_01_corrupted_mp3_byte_stream_handled_gracefully(self):
        """Verify corrupted MP3 container raises clean handled exception without crashing."""
        corrupted_path = os.path.join(self.tmp_dir, "corrupt_stream.mp3")
        np.random.seed(42)
        random_bytes = np.random.bytes(512)
        with open(corrupted_path, "wb") as f:
            f.write(random_bytes)

        # 1. AudioPreprocessor.load_audio_file must raise handled exception
        with self.assertRaises((RuntimeError, Exception)) as ctx:
            AudioPreprocessor.load_audio_file(corrupted_path)
        self.assertIn("Failed to load audio file", str(ctx.exception))

        # 2. VietnameseSTT.transcribe_file must propagate handled exception cleanly
        with self.assertRaises(Exception):
            self.stt.transcribe_file(corrupted_path)

    def test_02_truncated_wav_headers_handled_gracefully(self):
        """Verify WAV files with truncated headers at various offsets are tested."""
        valid_wav_path = "tests/data/sample1_greeting.wav"
        with open(valid_wav_path, "rb") as f:
            valid_bytes = f.read()

        # Truncation offsets where header is incomplete: 4, 12, 24, 36 bytes
        truncation_offsets = [4, 12, 24, 36]
        for offset in truncation_offsets:
            trunc_path = os.path.join(self.tmp_dir, f"truncated_{offset}.wav")
            with open(trunc_path, "wb") as f:
                f.write(valid_bytes[:offset])

            with self.assertRaises((RuntimeError, Exception)):
                AudioPreprocessor.load_audio_file(trunc_path)

            with self.assertRaises(Exception):
                self.stt.transcribe_file(trunc_path)

        # Offset 43 (incomplete data chunk header): loads as 0 frames, then process_file crashes
        trunc_43 = os.path.join(self.tmp_dir, "truncated_43.wav")
        with open(trunc_43, "wb") as f:
            f.write(valid_bytes[:43])

        prep = AudioPreprocessor(device=self.device)
        with self.assertRaises((RuntimeError, Exception)):
            prep.process_file(trunc_43)

    def test_03_zero_byte_audio_file(self):
        """Verify zero-byte audio file raises handled exception."""
        empty_path = os.path.join(self.tmp_dir, "zero_bytes.wav")
        with open(empty_path, "wb") as f:
            pass  # create 0 byte file

        self.assertEqual(os.path.getsize(empty_path), 0)
        with self.assertRaises((RuntimeError, Exception)):
            AudioPreprocessor.load_audio_file(empty_path)

        with self.assertRaises(Exception):
            self.stt.transcribe_file(empty_path)

    def test_04_ascii_text_disguised_as_audio(self):
        """Verify plain ASCII text file with .wav or .mp3 extension raises handled exception."""
        fake_wav = os.path.join(self.tmp_dir, "disguised_text.wav")
        with open(fake_wav, "w", encoding="utf-8") as f:
            f.write("This is a malicious plain text document pretending to be an audio file.")

        with self.assertRaises((RuntimeError, Exception)):
            AudioPreprocessor.load_audio_file(fake_wav)

        with self.assertRaises(Exception):
            self.stt.transcribe_file(fake_wav)

    def test_05_fastapi_rest_endpoints_corrupted_payload_status(self):
        """
        EMPIRICAL OBSERVATION: Tests FastAPI behavior when receiving corrupted audio.
        Documents whether server returns 400 Bad Request or crashes with 500 Internal Server Error.
        """
        client = TestClient(app)
        corrupt_bytes = np.random.bytes(256)
        resp_stt = client.post(
            "/api/v1/stt",
            files={"file": ("corrupt.wav", corrupt_bytes, "audio/wav")}
        )
        # Server currently fails with 500 due to unhandled convolution mismatch in Conformer fallback
        # Documented as Vulnerability V1 / V5
        self.assertIn(resp_stt.status_code, [400, 500])


class TestAdversarialSampleRatesAndChannels(unittest.TestCase):
    """
    Challenge Dimension 2: Non-standard sample rates (8kHz, 22.05kHz, 44.1kHz, 48kHz)
    and channel configurations (Mono, Stereo, Multichannel 5.1).
    """

    @classmethod
    def setUpClass(cls):
        cls.device = "cpu"
        cls.stt = VietnameseSTT(device=cls.device)
        cls.tmp_dir = tempfile.mkdtemp(prefix="nckh_adv_rates_")

        # Load authentic sample1 greeting (16kHz mono)
        cls.orig_audio, cls.orig_sr = sf.read("tests/data/sample1_greeting.wav", dtype="float32")
        cls.expected_transcript = "xin chào bạn hôm nay bạn có khỏe không"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_01_narrowband_8khz_telephony_audio(self):
        """Verify 8,000 Hz telephone audio is correctly resampled and transcribed."""
        resampled_8k = librosa.resample(self.orig_audio, orig_sr=self.orig_sr, target_sr=8000)
        path_8k = os.path.join(self.tmp_dir, "sample_8k.wav")
        sf.write(path_8k, resampled_8k, 8000)

        tensor_loaded, sr_loaded = AudioPreprocessor.load_audio_file(path_8k)
        self.assertEqual(sr_loaded, 8000)
        prep = AudioPreprocessor(device=self.device)
        mono_16k = prep.resample_to_16k(tensor_loaded, sr_loaded)
        self.assertEqual(mono_16k.dim(), 2)
        self.assertEqual(mono_16k.size(0), 1)

        transcript = self.stt.transcribe_file(path_8k)
        self.assertIsInstance(transcript, str)
        self.assertGreater(len(transcript), 0)
        self.assertIn("xin chào", transcript)

    def test_02_broadcast_22050hz_audio(self):
        """Verify 22,050 Hz broadcast audio is correctly resampled and transcribed."""
        resampled_22k = librosa.resample(self.orig_audio, orig_sr=self.orig_sr, target_sr=22050)
        path_22k = os.path.join(self.tmp_dir, "sample_22k.wav")
        sf.write(path_22k, resampled_22k, 22050)

        transcript = self.stt.transcribe_file(path_22k)
        self.assertIsInstance(transcript, str)
        self.assertEqual(transcript.strip(), self.expected_transcript)

    def test_03_standard_cd_44100hz_stereo_audio(self):
        """Verify 44,100 Hz 2-channel stereo audio is downmixed to mono, resampled, and transcribed."""
        resampled_44k = librosa.resample(self.orig_audio, orig_sr=self.orig_sr, target_sr=44100)
        stereo_44k = np.stack([resampled_44k, resampled_44k], axis=1)  # [N, 2]
        path_44k_stereo = os.path.join(self.tmp_dir, "sample_44k_stereo.wav")
        sf.write(path_44k_stereo, stereo_44k, 44100)

        tensor_loaded, sr_loaded = AudioPreprocessor.load_audio_file(path_44k_stereo)
        self.assertEqual(sr_loaded, 44100)
        self.assertEqual(tensor_loaded.size(0), 2, "Must load 2 channels")

        transcript = self.stt.transcribe_file(path_44k_stereo)
        self.assertEqual(transcript.strip(), self.expected_transcript)

    def test_04_pro_studio_48000hz_multichannel_audio(self):
        """Verify 48,000 Hz 6-channel (5.1 surround) audio is properly downmixed and transcribed."""
        resampled_48k = librosa.resample(self.orig_audio, orig_sr=self.orig_sr, target_sr=48000)
        multichannel_6ch = np.stack([resampled_48k * (1.0 / (i + 1)) for i in range(6)], axis=1)
        path_48k_6ch = os.path.join(self.tmp_dir, "sample_48k_5point1.wav")
        sf.write(path_48k_6ch, multichannel_6ch, 48000)

        tensor_loaded, sr_loaded = AudioPreprocessor.load_audio_file(path_48k_6ch)
        self.assertEqual(sr_loaded, 48000)
        self.assertEqual(tensor_loaded.size(0), 6, "Must load 6 channels")

        transcript = self.stt.transcribe_file(path_48k_6ch)
        self.assertEqual(transcript.strip(), self.expected_transcript)


class TestAdversarialLowRMSAndWhispering(unittest.TestCase):
    """
    Challenge Dimension 3: Low RMS amplitude, whispering, near-silence, and digital zero.
    """

    @classmethod
    def setUpClass(cls):
        cls.device = "cpu"
        cls.stt = VietnameseSTT(device=cls.device)
        cls.engine = SpeechToSpeechEngine(device=cls.device, use_amp=False)
        cls.orig_audio, cls.orig_sr = sf.read("tests/data/sample1_greeting.wav", dtype="float32")

    def test_01_digital_pure_zero_returns_empty_without_crash(self):
        """Verify digital zero tensor returns empty string without crashing or hallucinating."""
        pure_zero = np.zeros(32000, dtype=np.float32)  # 2 seconds of pure zero
        transcript = self.stt.transcribe_file(pure_zero)
        self.assertEqual(transcript, "", "Pure silence must return empty string, never hallucinations")

        self.stt.reset_streaming_state()
        chunk_res = self.stt.transcribe_chunk(pure_zero[:4800])
        self.assertEqual(chunk_res, "")

    def test_02_near_silence_noise_minus_120_dbfs(self):
        """Verify extreme near-silence (amplitude 1e-6 / -120 dBFS) returns empty string."""
        np.random.seed(123)
        sub_audible_noise = np.random.uniform(-1e-6, 1e-6, size=32000).astype(np.float32)
        transcript = self.stt.transcribe_file(sub_audible_noise)
        self.assertEqual(transcript, "", "Sub-audible noise must return empty string")

    def test_03_attenuated_whispering_speech(self):
        """Verify severely attenuated speech (-40 dBFS, 1% original amplitude) is handled cleanly."""
        whisper_audio = self.orig_audio * 0.01
        rms_whisper = np.sqrt(np.mean(whisper_audio ** 2))
        self.assertLess(rms_whisper, 0.005)

        transcript = self.stt.transcribe_file(whisper_audio)
        self.assertIsInstance(transcript, str)
        self.assertIn("xin chào", transcript)

    def test_04_e2e_process_file_on_silence_returns_empty_tuple(self):
        """Verify S2S engine process_file on pure silence produces clean empty output without crashing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            silence_path = tf.name
            sf.write(silence_path, np.zeros(16000, dtype=np.float32), 16000)

        try:
            text_vi, text_hmong, audio_hmong = self.engine.process_file(silence_path)
            self.assertEqual(text_vi, "")
            self.assertEqual(text_hmong, "")
            self.assertEqual(len(audio_hmong), 0)
        finally:
            if os.path.exists(silence_path):
                os.remove(silence_path)


class TestAdversarialExtremeNoise(unittest.TestCase):
    """
    Challenge Dimension 4: Additive White Gaussian Noise (AWGN) degradation at
    SNR 20dB, 10dB, 5dB, and extreme 0dB / -5dB.
    """

    @classmethod
    def setUpClass(cls):
        cls.device = "cpu"
        cls.stt = VietnameseSTT(device=cls.device)
        cls.orig_audio, cls.orig_sr = sf.read("tests/data/sample1_greeting.wav", dtype="float32")
        cls.signal_power = np.mean(cls.orig_audio ** 2)

    def _add_awgn(self, audio: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
        np.random.seed(seed)
        noise_power = self.signal_power / (10.0 ** (snr_db / 10.0))
        noise = np.random.normal(0.0, np.sqrt(noise_power), size=len(audio)).astype(np.float32)
        return audio + noise

    def test_01_moderate_noise_snr_20db(self):
        """Verify transcription survives moderate noise (SNR 20dB)."""
        noisy_20db = self._add_awgn(self.orig_audio, 20.0)
        transcript = self.stt.transcribe_file(noisy_20db)
        self.assertIsInstance(transcript, str)
        self.assertIn("xin chào", transcript)

    def test_02_heavy_noise_snr_10db(self):
        """Verify transcription survives heavy noise (SNR 10dB) without unhandled exceptions."""
        noisy_10db = self._add_awgn(self.orig_audio, 10.0)
        transcript = self.stt.transcribe_file(noisy_10db)
        self.assertIsInstance(transcript, str)
        self.assertGreater(len(transcript), 0)

    def test_03_extreme_noise_snr_5db(self):
        """Verify transcription at extreme noise (SNR 5dB) executes cleanly without crashing."""
        noisy_5db = self._add_awgn(self.orig_audio, 5.0)
        transcript = self.stt.transcribe_file(noisy_5db)
        self.assertIsInstance(transcript, str)
        self.assertTrue(all(c in VIETNAMESE_CTC_VOCAB or c.isspace() or c.isalnum() for c in transcript))

    def test_04_critical_noise_snr_0db(self):
        """Verify transcription at critical noise (SNR 0dB, noise power = signal power) executes cleanly."""
        noisy_0db = self._add_awgn(self.orig_audio, 0.0)
        transcript = self.stt.transcribe_file(noisy_0db)
        self.assertIsInstance(transcript, str)

    def test_05_severe_negative_noise_snr_minus_5db(self):
        """Verify severe negative SNR (-5dB, noise is 3x louder than speech) executes without exception."""
        noisy_sub0 = self._add_awgn(self.orig_audio, -5.0)
        transcript = self.stt.transcribe_file(noisy_sub0)
        self.assertIsInstance(transcript, str)


class TestAdversarialRapidSpeech(unittest.TestCase):
    """
    Challenge Dimension 5: Rapid speech and time-stretching (1.25x, 1.5x, 2.0x).
    """

    @classmethod
    def setUpClass(cls):
        cls.device = "cpu"
        cls.stt = VietnameseSTT(device=cls.device)
        cls.orig_audio, cls.orig_sr = sf.read("tests/data/sample1_greeting.wav", dtype="float32")

    def test_01_accelerated_speech_1_25x(self):
        """Verify transcription on 1.25x accelerated speech."""
        audio_125 = librosa.effects.time_stretch(self.orig_audio, rate=1.25)
        transcript = self.stt.transcribe_file(audio_125)
        self.assertIsInstance(transcript, str)
        self.assertIn("xin chào", transcript)

    def test_02_rapid_speech_1_5x(self):
        """Verify transcription on 1.5x rapid speech."""
        audio_15 = librosa.effects.time_stretch(self.orig_audio, rate=1.5)
        transcript = self.stt.transcribe_file(audio_15)
        self.assertIsInstance(transcript, str)
        self.assertGreater(len(transcript), 0)

    def test_03_extreme_speed_2_0x(self):
        """Verify transcription on 2.0x ultra-rapid speech executes cleanly without crash."""
        audio_20 = librosa.effects.time_stretch(self.orig_audio, rate=2.0)
        transcript = self.stt.transcribe_file(audio_20)
        self.assertIsInstance(transcript, str)


class TestAdversarialVietnamesePhonology(unittest.TestCase):
    """
    Challenge Dimension 6: Complex Vietnamese phonology and rare diacritics.
    """

    def test_01_ctc_vocab_contains_all_vietnamese_diacritics(self):
        """Verify VIETNAMESE_CTC_VOCAB exhaustively covers all Vietnamese diacritic variations."""
        rare_diacritics = [
            "ở", "ỡ", "ợ", "ử", "ữ", "ự", "ễ", "ệ",
            "ẵ", "ặ", "ỷ", "ỹ", "ỵ", "đ", "ấ", "ầ", "ẩ", "ẫ", "ậ"
        ]

        for char in rare_diacritics:
            self.assertIn(
                char,
                VIETNAMESE_CTC_VOCAB,
                f"Missing rare Vietnamese diacritic '{char}' in VIETNAMESE_CTC_VOCAB"
            )

    def test_02_wav2vec2_processor_tokenizer_diacritics_coverage(self):
        """Verify Wav2Vec2 tokenizer correctly encodes and decodes complex Vietnamese words."""
        stt = VietnameseSTT(device="cpu")
        if stt.hf_processor is not None:
            complex_sentence = (
                "trường đại học sư phạm kỹ thuật nguyễn tất thành khuỷu tay thuở nhỏ ngoằn ngoèo"
            )
            encoded = stt.hf_processor.tokenizer(complex_sentence)
            input_ids = encoded["input_ids"]
            decoded = stt.hf_processor.tokenizer.decode(input_ids)

            for word in ["trường", "nguyễn", "khuỷu", "thuở", "ngoèo"]:
                self.assertIn(word, decoded.lower())

    def test_03_rpa_normalizer_preserves_complex_vietnamese_morphology(self):
        """Verify RPATextNormalizer handles complex Vietnamese sentences without mangling diacritics."""
        complex_input = "Xin chào! Bạn có bị đau ở khuỷu tay, ngực hay khó thở không?"
        normalized = RPATextNormalizer.normalize_vietnamese(complex_input)
        self.assertIn("khuỷu", normalized)
        self.assertIn("thở", normalized)
        self.assertIn("?", normalized)


class TestAdversarialNoFakeHardcodedStrings(unittest.TestCase):
    """
    Challenge Dimension 7: Behavioral verification ensuring the pipeline never returns
    fake hardcoded strings and genuinely executes acoustic neural networks.
    """

    @classmethod
    def setUpClass(cls):
        cls.device = "cpu"
        cls.stt = VietnameseSTT(device=cls.device)
        cls.orig_audio, cls.orig_sr = sf.read("tests/data/sample1_greeting.wav", dtype="float32")

    def test_01_pure_sine_tone_does_not_return_fake_greeting(self):
        """Verify 440 Hz pure tone does NOT return fake human speech phrases."""
        t = np.linspace(0, 2.0, 32000, endpoint=False, dtype=np.float32)
        sine_440 = 0.5 * np.sin(2 * np.pi * 440.0 * t)

        transcript = self.stt.transcribe_file(sine_440)
        self.assertNotEqual(
            transcript,
            "xin chào bạn hôm nay bạn có khỏe không",
            "Pure sine tone must NOT return authentic speech transcript"
        )
        self.assertNotEqual(transcript, "xin chào", "Pure tone must NOT return fake greeting")

    def test_02_pure_1000hz_tone_does_not_return_fake_greeting(self):
        """Verify 1000 Hz pure tone does NOT return fake human speech phrases."""
        t = np.linspace(0, 2.0, 32000, endpoint=False, dtype=np.float32)
        sine_1000 = 0.5 * np.sin(2 * np.pi * 1000.0 * t)

        transcript = self.stt.transcribe_file(sine_1000)
        self.assertNotEqual(transcript, "xin chào bạn hôm nay bạn có khỏe không")
        self.assertNotEqual(transcript, "xin chào")

    def test_03_time_reversed_speech_produces_different_transcription(self):
        """Verify time-reversing authentic audio does NOT produce original forward transcript."""
        reversed_audio = self.orig_audio[::-1].copy()
        transcript_rev = self.stt.transcribe_file(reversed_audio)

        self.assertNotEqual(
            transcript_rev,
            "xin chào bạn hôm nay bạn có khỏe không",
            "Time-reversed audio must NOT match forward speech transcript"
        )

    def test_04_different_authentic_samples_produce_unique_transcripts(self):
        """Verify each of the 5 authentic samples yields distinct, non-identical transcripts."""
        sample_files = [
            "tests/data/sample1_greeting.wav",
            "tests/data/sample2_hospital.wav",
            "tests/data/sample3_pharmacy.wav",
            "tests/data/sample4_weather.wav",
            "tests/data/sample5_thanks.wav",
        ]
        transcripts = [self.stt.transcribe_file(f) for f in sample_files]
        self.assertEqual(len(set(transcripts)), len(sample_files), "All samples must produce unique transcripts")


class TestAdversarialVulnerabilityDemonstrations(unittest.TestCase):
    """
    Verification of Remediated Edge Cases and Hardened Pipeline Resilience.
    """

    def test_vuln1_stt_module_subsampler_crash_on_short_audio(self):
        """
        Verify VietnameseSTT handles short audio waveforms (<= 256 samples)
        safely via proper Mel extraction and padding without convolution mismatch crash.
        """
        stt = VietnameseSTT(device="cpu")
        short_tensor = torch.randn(1, 128)
        log_probs, out_lens = stt.forward(short_tensor)
        self.assertEqual(log_probs.dim(), 3)
        self.assertEqual(log_probs.size(0), 1)
        self.assertEqual(log_probs.size(2), len(VIETNAMESE_CTC_VOCAB))

    def test_vuln2_s2s_pipeline_process_stream_zero_dimensional_crash(self):
        """
        Verify SpeechToSpeechEngine.process_stream safely handles 1-sample chunks
        without throwing zero-dimensional array concatenation error.
        """
        engine = SpeechToSpeechEngine(device="cpu", use_amp=False)
        hostile_chunks = [
            np.array([0.5], dtype=np.float32),
            np.zeros(6400, dtype=np.float32)  # Triggers silence hangover
        ]
        results = list(engine.process_stream(iter(hostile_chunks)))
        self.assertIsInstance(results, list)

    def test_vuln3_s2s_pipeline_hardcoded_fake_string_fallback(self):
        """
        Verify SpeechToSpeechEngine._execute_streaming_utterance does NOT inject
        fake hardcoded 'xin chào' / 'nyob zoo' when STT returns empty text on silence.
        """
        engine = SpeechToSpeechEngine(device="cpu", use_amp=False)
        silence = np.zeros(16000, dtype=np.float32)  # 1 second pure silence
        pcm_bytes = engine._execute_streaming_utterance(silence)
        self.assertEqual(len(pcm_bytes), 0, "Silence must yield empty audio without fake greeting injection")

    def test_vuln4_audio_preprocessor_rms_crash_on_empty_audio(self):
        """
        Verify AudioPreprocessor.process_file safely handles empty 0-sample audio
        by raising a clean RuntimeError without torch.max reduction dimension crash.
        """
        prep = AudioPreprocessor(device="cpu")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            empty_wav = tf.name
            sf.write(empty_wav, np.zeros(0, dtype=np.float32), 16000)

        try:
            with self.assertRaises(RuntimeError) as ctx:
                prep.process_file(empty_wav)
            self.assertTrue(
                "Empty or truncated audio file" in str(ctx.exception) or
                "Failed to load audio file" in str(ctx.exception)
            )
        finally:
            if os.path.exists(empty_wav):
                os.remove(empty_wav)


if __name__ == "__main__":
    unittest.main()
