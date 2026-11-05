"""
tests/test_m5_eval.py
Unit and Integration Test Suite for Milestone M5: Academic Evaluation & Benchmark Suite.
Tests:
  1. compute_stt_metrics (WER, CER via jiwer).
  2. compute_mt_metrics (BLEU-4, SacreBLEU, chrF++ via sacrebleu).
  3. create_synthetic_speech_wav (16kHz audio generation and container validation).
  4. AcademicEvaluator model loading and configuration parsing.
  5. prepare_dataset (self-contained benchmark generation in dry_run mode).
  6. generate_latex_table (booktabs compliance: \\toprule, \\midrule, \\bottomrule).
  7. End-to-end evaluation execution and latency percentile computations (P50, P90, P99, Mean).
"""

import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scripts.eval import (
    compute_stt_metrics,
    compute_mt_metrics,
    create_synthetic_speech_wav,
    generate_latex_table,
    AcademicEvaluator,
    SYNTHETIC_BENCHMARK_CORPUS
)


class TestM5EvaluationSuite(unittest.TestCase):
    """Test suite covering scripts/eval.py components."""

    def test_stt_metrics_perfect_and_imperfect(self):
        """Verify WER and CER calculations on identical and divergent transcripts."""
        refs = ["xin chào bạn hôm nay", "tôi là người việt nam"]
        # Perfect match
        wer_0, cer_0 = compute_stt_metrics(refs, refs)
        self.assertAlmostEqual(wer_0, 0.0, places=4)
        self.assertAlmostEqual(cer_0, 0.0, places=4)

        # Mismatched transcripts
        hyps = ["xin chào bạn hôm qua", "tôi là người h mông"]
        wer_sub, cer_sub = compute_stt_metrics(refs, hyps)
        self.assertGreater(wer_sub, 0.0)
        self.assertGreater(cer_sub, 0.0)

    def test_mt_metrics_perfect_and_imperfect(self):
        """Verify SacreBLEU, BLEU-4, and chrF++ scores."""
        refs = ["nyob zoo koj hnub no koj puas nyob zoo", "kuv yog neeg hmoob"]
        # Identical hypothesis should achieve maximum scores
        bleu4, sacre, chrf = compute_mt_metrics(refs, refs)
        self.assertGreater(bleu4, 80.0)
        self.assertGreater(sacre, 80.0)
        self.assertGreater(chrf, 80.0)

        # Diverse hypothesis
        divergent_hyps = ["koj puas nyob zoo hnub no", "kuv yog neeg Nyab Laj"]
        bleu4_div, sacre_div, chrf_div = compute_mt_metrics(refs, divergent_hyps)
        self.assertLess(bleu4_div, bleu4)
        self.assertGreater(chrf_div, 0.0)

    def test_synthetic_speech_wav_generation(self):
        """Verify synthetic WAV generator produces valid 16kHz mono audio container."""
        dur = 0.5
        wav_path, wav_np = create_synthetic_speech_wav(duration_s=dur, sample_rate=16000)
        try:
            self.assertTrue(os.path.exists(wav_path))
            self.assertGreater(os.path.getsize(wav_path), 44)  # Greater than WAV header

            with wave.open(wav_path, "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)  # 16-bit
                self.assertEqual(wf.getframerate(), 16000)
                n_frames = wf.getnframes()
                self.assertEqual(n_frames, int(dur * 16000))

            self.assertIsInstance(wav_np, np.ndarray)
            self.assertEqual(len(wav_np), int(dur * 16000))
            self.assertTrue(np.all(wav_np >= -1.0) and np.all(wav_np <= 1.0))
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    def test_academic_evaluator_initialization(self):
        """Verify AcademicEvaluator instantiates all pipeline modules."""
        evaluator = AcademicEvaluator(device="cpu", beam_size=2)
        self.assertIsNotNone(evaluator.preprocessor)
        self.assertIsNotNone(evaluator.stt)
        self.assertIsNotNone(evaluator.mt)
        self.assertIsNotNone(evaluator.tts)
        self.assertIsNotNone(evaluator.s2s)
        self.assertEqual(evaluator.device_str, "cpu")

    def test_prepare_dataset_dry_run(self):
        """Verify dataset preparation in dry_run mode produces requested samples."""
        evaluator = AcademicEvaluator(device="cpu", beam_size=2)
        num_requested = 3
        samples = evaluator.prepare_dataset(num_samples=num_requested, dry_run=True)
        try:
            self.assertEqual(len(samples), num_requested)
            for s in samples:
                self.assertIn("audio_path", s)
                self.assertIn("text_vi_ref", s)
                self.assertIn("text_hmong_ref", s)
                self.assertTrue(os.path.exists(s["audio_path"]))
        finally:
            # Cleanup created temporary files
            for s in samples:
                if s.get("is_temp_file") and os.path.exists(s["audio_path"]):
                    os.remove(s["audio_path"])

    def test_latex_table_generation_and_booktabs(self):
        """Verify LaTeX generator creates valid booktabs table."""
        sample_results = {
            "metadata": {
                "num_samples": 5,
                "device": "cpu",
                "timestamp": "2026-10-19T00:00:00Z"
            },
            "stt": {
                "wer_percent": 15.2,
                "cer_percent": 6.8,
                "latency_mean_ms": 110.5,
                "latency_p50_ms": 105.0,
                "target_wer_percent": 20.0,
                "target_cer_percent": 10.0
            },
            "mt": {
                "bleu_4": 35.4,
                "sacrebleu": 36.1,
                "chrf_pp": 58.2,
                "latency_mean_ms": 85.0,
                "target_sacrebleu": 25.0,
                "target_chrf": 45.0
            },
            "tts": {
                "rtf": 4.5,
                "latency_mean_ms": 120.0,
                "audio_duration_mean_s": 1.45,
                "target_rtf": 1.5
            },
            "s2s": {
                "latency_mean_ms": 320.5,
                "latency_std_ms": 45.2,
                "latency_p50_ms": 310.0,
                "latency_p90_ms": 380.0,
                "latency_p99_ms": 420.0,
                "rtf": 4.5,
                "target_compliance_pct": 100.0,
                "target_latency_ms": 1500.0,
                "decomposition": {
                    "preprocess_mean_ms": 15.0,
                    "stt_mean_ms": 110.5,
                    "mt_mean_ms": 85.0,
                    "tts_mean_ms": 110.0,
                    "total_mean_ms": 320.5
                }
            }
        }

        tf = tempfile.NamedTemporaryFile(suffix=".tex", delete=False)
        tex_path = tf.name
        tf.close()

        try:
            tex_str = generate_latex_table(sample_results, output_path=tex_path)
            self.assertTrue(os.path.exists(tex_path))
            self.assertGreater(os.path.getsize(tex_path), 500)

            # Check for booktabs compliance
            self.assertIn("\\toprule", tex_str)
            self.assertIn("\\midrule", tex_str)
            self.assertIn("\\bottomrule", tex_str)

            # Check for metric mentions
            self.assertIn("Word Error Rate (WER)", tex_str)
            self.assertIn("SacreBLEU", tex_str)
            self.assertIn("Real-Time Factor (RTF)", tex_str)
            self.assertIn("Latency P50", tex_str)
            self.assertIn("Latency P90", tex_str)
            self.assertIn("Latency P99", tex_str)
        finally:
            if os.path.exists(tex_path):
                os.remove(tex_path)

    def test_e2e_evaluation_run(self):
        """Verify genuine evaluation pass on 2 dry_run samples."""
        evaluator = AcademicEvaluator(device="cpu", beam_size=1)
        samples = evaluator.prepare_dataset(num_samples=2, dry_run=True)
        results = evaluator.evaluate(samples)

        self.assertIn("stt", results)
        self.assertIn("mt", results)
        self.assertIn("tts", results)
        self.assertIn("s2s", results)

        # Verify percentile outputs
        s2s = results["s2s"]
        self.assertIn("latency_p50_ms", s2s)
        self.assertIn("latency_p90_ms", s2s)
        self.assertIn("latency_p99_ms", s2s)
        self.assertIn("latency_mean_ms", s2s)
        self.assertIn("latency_std_ms", s2s)
        self.assertIn("rtf", s2s)
        self.assertIn("target_compliance_pct", s2s)


if __name__ == "__main__":
    unittest.main()
