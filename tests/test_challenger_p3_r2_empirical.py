"""
tests/test_challenger_p3_r2_empirical.py
Empirical Adversarial Challenge Suite by challenger_p3_r2_2.
Stress-testing TTS vocoder continuous synthesis, RMS loudness corridor, speed scaling,
thread safety, boundary edge cases, and HTML5 DOM playback handling.
"""

import io
import os
import sys
import time
import wave
import warnings
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.tts_module import HmongVITS
from src.api.app_stream import app, get_engine, INDEX_HTML


class TestChallengerP3R2AcousticStress(unittest.TestCase):
    """Adversarial stress-testing of HmongVITS across boundary conditions and extreme speeds."""

    @classmethod
    def setUpClass(cls):
        cls.tts = HmongVITS(sample_rate=16000, device="cpu")

    def test_01_extreme_duration_and_compound_narratives(self):
        """Stress-test: Super-long utterances (>100 words, >300,000 samples) to verify zero boundary distortion."""
        long_texts = [
            # 50 words
            "thaum ub muaj ib tug txiv ntxawm nyob rau hauv lub zos ntawd nws muaj ntau tus me nyuam heev thiab nws ua teb hnyav heev txhua hnub tsis muaj hnub so li tab sis nws tsev neeg zoo siab heev vim lawv muaj mov noj txaus thiab muaj tsev nyob sov siab heev tiag tiag",
            # 75 words
            "peb cov hmoob nyob rau toj siab muaj ntau yam kab lis kev cai zoo nkauj thiab qub txeeg qub teg heev uas tsim nyog ceev tseg rau peb cov me nyuam tiam tom ntej kom lawv paub txog lawv lub hauv paus thiab lawv cov poj koob yawm txwv uas tau ua neej nyob nyuaj xeeb heev yav thaum ub thaum lawv tsiv ntawm teb chaws suav tuaj nyob rau sab pem teb chaws nplog thiab nyab laj",
            # 100+ words
            "yog tias koj xav mus kawm ntawv nyob rau lub nroog loj koj yuav tsum mob siab kawm ntawv kom tiav thiab rov qab tuaj pab peb lub zos kom peb lub zos vam meej tuaj thiab txhob hnov qab txog peb cov lus hmoob peb cov kab lis kev cai thiab peb txoj kev sib hlub sib pab ntawm cov kwv tij hmoob " * 2
        ]

        for idx, text in enumerate(long_texts, 1):
            t0 = time.perf_counter()
            audio = self.tts.synthesize(text)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            self.assertGreater(len(audio), 50000, f"Long text #{idx} audio too short")
            rms = float(np.sqrt(np.mean(audio ** 2)))
            dbfs = 20.0 * np.log10(rms) if rms > 0 else -999.0
            peak = float(np.max(np.abs(audio)))
            dc = float(np.mean(audio))

            print(f"[Long #{idx}] samples={len(audio)}, duration={len(audio)/16000:.2f}s, dbfs={dbfs:.2f}, peak={peak:.4f}, dc={dc:.8f}, latency={latency_ms:.1f}ms")

            # Check strict RMS corridor [-20.0, -16.0] dBFS
            self.assertGreaterEqual(dbfs, -20.0, f"Long text #{idx} too quiet: {dbfs:.2f} dBFS")
            self.assertLessEqual(dbfs, -16.0, f"Long text #{idx} too loud: {dbfs:.2f} dBFS")
            self.assertLessEqual(peak, 0.99, f"Long text #{idx} clipped: peak {peak:.4f}")
            self.assertLess(abs(dc), 1e-3, f"Long text #{idx} excessive DC: {dc:.6f}")

    def test_02_extreme_speed_scaling_bounds(self):
        """Stress-test: Extreme speed scaling from 0.25x to 3.5x across diverse lengths."""
        speeds = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        test_sentences = [
            "kuv nyob zoo",
            "hnub no huab cua zoo heev peb yuav mus tom khw yuav nqaij npuas",
            "tus kws kho mob hais tias koj yuav tsum tau noj tshuaj raws sij hawm"
        ]

        for s_idx, sentence in enumerate(test_sentences, 1):
            prev_len = None
            for spd in speeds:
                audio = self.tts.synthesize(sentence, speed=spd)
                self.assertGreater(len(audio), 0, f"Speed {spd} on sentence #{s_idx} produced 0 samples")
                self.assertFalse(np.isnan(audio).any(), f"Speed {spd} produced NaNs")
                self.assertFalse(np.isinf(audio).any(), f"Speed {spd} produced Infs")

                rms = float(np.sqrt(np.mean(audio ** 2)))
                dbfs = 20.0 * np.log10(rms) if rms > 0 else -999.0
                peak = float(np.max(np.abs(audio)))
                dc = float(np.mean(audio))

                # Duration monotonicity: faster speed should generally result in shorter or equal sample count
                if prev_len is not None:
                    self.assertLessEqual(
                        len(audio), prev_len + 16, # allow small rounding margin
                        f"Sentence #{s_idx} at speed {spd} produced {len(audio)} > prev {prev_len}"
                    )
                prev_len = len(audio)

                # Loudness & safety constraints
                self.assertGreaterEqual(dbfs, -20.0, f"Sentence #{s_idx} speed {spd} too quiet: {dbfs:.2f} dBFS")
                self.assertLessEqual(dbfs, -16.0, f"Sentence #{s_idx} speed {spd} too loud: {dbfs:.2f} dBFS")
                self.assertLessEqual(peak, 0.99, f"Sentence #{s_idx} speed {spd} clipped: peak {peak:.4f}")
                self.assertLess(abs(dc), 1e-3, f"Sentence #{s_idx} speed {spd} DC: {dc:.6f}")

    def test_03_boundary_and_degenerate_inputs(self):
        """Stress-test: Empty, single char, pure punctuation, numbers, non-RPA characters."""
        edge_cases = [
            "",
            "   \t\n  ",
            "!",
            "?...,,,",
            "12345",
            "@#$%^&*",
            "a",
            "z",
            "ñ",
            "😀"
        ]

        for raw_in in edge_cases:
            audio = self.tts.synthesize(raw_in)
            self.assertIsInstance(audio, np.ndarray)
            self.assertEqual(audio.dtype, np.float32)
            # If audio is generated, it must be valid float in [-1.0, 1.0] and have no NaNs
            if len(audio) > 0:
                self.assertFalse(np.isnan(audio).any())
                self.assertFalse(np.isinf(audio).any())
                peak = float(np.max(np.abs(audio)))
                self.assertLessEqual(peak, 0.99)
                dc = float(np.mean(audio))
                self.assertLess(abs(dc), 1e-3)
                rms = float(np.sqrt(np.mean(audio ** 2)))
                dbfs = 20.0 * np.log10(rms) if rms > 0 else -999.0
                # If non-silent, must be in corridor
                if rms > 1e-4:
                    self.assertGreaterEqual(dbfs, -20.0)
                    self.assertLessEqual(dbfs, -16.0)

    def test_04_thread_safety_concurrent_synthesis(self):
        """Stress-test: 20 concurrent threads calling synthesize() simultaneously on shared HmongVITS instance."""
        prompts = [
            "nyob zoo",
            "ua tsaug",
            "kuv nyob zoo heev",
            "koj tuaj qhov twg tuaj",
            "hnub no huab cua zoo heev",
            "kuv xav mus tom khw yuav nqaij npuas",
            "thov muab ib khob dej txias rau kuv",
            "tsev kho mob nyob qhov twg thov pab qhia",
            "ib ob peb plaub tsib rau xya yim cuaj kaum",
            "cais txiav nres koj puas to taub zoo",
        ] * 2  # 20 concurrent jobs

        def run_synth(item):
            idx, text = item
            out = self.tts.synthesize(text)
            wav_bytes = self.tts.synthesize_to_bytes(text, format="wav")
            return idx, text, len(out), len(wav_bytes)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(run_synth, (i, p)) for i, p in enumerate(prompts)]
            for fut in as_completed(futures):
                idx, text, n_samples, n_bytes = fut.result()
                self.assertGreater(n_samples, 0, f"Thread {idx} produced 0 samples")
                self.assertGreater(n_bytes, 44, f"Thread {idx} produced incomplete WAV")


class TestChallengerP3R2APISafety(unittest.TestCase):
    """Adversarial stress-testing of FastAPI API concurrency and payload edge cases."""

    @classmethod
    def setUpClass(cls):
        cls.engine = get_engine(device="cpu", use_amp=False)
        app.state.engine = cls.engine
        cls.client = TestClient(app)

    def test_01_high_concurrency_t2s_stress(self):
        """Send 25 rapid concurrent requests to POST /api/v1/t2s."""
        sentences = [
            f"Câu thử nghiệm áp lực số {i}: bác sĩ ơi tôi cần tư vấn về sức khỏe gia đình"
            for i in range(25)
        ]

        def call_t2s(s):
            r = self.client.post("/api/v1/t2s", json={"text": s, "speed": 1.0})
            return s, r.status_code, len(r.content), r.headers

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(call_t2s, s) for s in sentences]
            for fut in as_completed(futures):
                s, status, length, headers = fut.result()
                self.assertEqual(status, 200, f"T2S failed for sentence '{s}'")
                self.assertGreater(length, 44)
                self.assertEqual(headers.get("content-type"), "audio/wav")
                self.assertIn("x-vi-text", headers)
                self.assertIn("x-hmong-text", headers)

    def test_02_high_concurrency_s2s_dual_input_stress(self):
        """Send 20 rapid concurrent requests to POST /api/v1/s2s with direct text."""
        sentences = [
            f"Thử nghiệm đầu vào trực tiếp {i}: hôm nay trời nhiều mây và có thể mưa dông"
            for i in range(20)
        ]

        def call_s2s(i, s):
            if i % 2 == 0:
                r = self.client.post("/api/v1/s2s", json={"text": s})
            else:
                r = self.client.post("/api/v1/s2s", data={"text": s})
            return i, s, r.status_code, len(r.content)

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(call_s2s, i, s) for i, s in enumerate(sentences)]
            for fut in as_completed(futures):
                i, s, status, length = fut.result()
                self.assertEqual(status, 200, f"S2S direct text failed #{i}: '{s}'")
                self.assertGreater(length, 44)


class TestChallengerP3R2DOMVerification(unittest.TestCase):
    """Adversarial audit of HTML5 DOM structure and autoplay mitigation scripts."""

    def test_01_comprehensive_autoplay_handling_logic(self):
        """Verify audio playback promise rejection handling and UI indicators in INDEX_HTML."""
        html = INDEX_HTML
        # 1. Autoplay catch handler
        self.assertIn(".catch(", html)
        self.assertIn("autoplayNotice", html)
        self.assertIn("btn-replay-pulse", html)

        # 2. Both nativeAudioPlayer and replayBtn are present
        self.assertIn('id="nativeAudioPlayer"', html)
        self.assertIn('id="replayBtn"', html)
        self.assertIn('id="autoplayNotice"', html)

        # 3. User interaction gesture unlock
        self.assertIn("unlockAudioPlayback", html)
        self.assertIn("AudioContext", html)

        # 4. Dual input elements
        self.assertIn('id="viTextInput"', html)
        self.assertIn('id="translateTextBtn"', html)
        self.assertIn('id="clearTextBtn"', html)


if __name__ == "__main__":
    unittest.main()
