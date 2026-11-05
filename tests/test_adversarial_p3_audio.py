"""
tests/test_adversarial_p3_audio.py
Adversarial Stress Test Suite for Requirements R1 (Dual Input) and R3 (Audible TTS Waveform & Browser Playback).
Milestone P3-M5 Consortium Challenge Verification.

Challenge Vectors:
  1. VITS synthesis across 36 diverse RPA sentences (1-syllable to 29-word compound narrative sentences):
     - Synthesized RMS amplitude strictly within [-20.0, -16.0] dBFS.
     - Peak amplitude strictly <= 0.99 (zero digital clipping).
     - DC offset strictly < 1e-3.
     - Output WAV bytes contain valid standard 44-byte RIFF headers decodable by soundfile without warnings.
     - Synthesis speed robustness across 0.5x, 1.0x, 1.5x, and 2.5x.
  2. Dual Input & API Concurrency Stress:
     - Rapid direct text payloads to POST /api/v1/t2s and POST /api/v1/s2s.
     - Concurrent audio file uploads (MP3 & WAV) to POST /api/v1/s2s.
     - Mixed concurrent stress (MP3, WAV, T2S text, S2S text) testing thread safety and zero crashes.
     - Edge-case payloads (empty string, whitespace, unicode/emojis, massive text).
  3. HTML5 Player DOM & Autoplay Mitigation:
     - DOM elements: #nativeAudioPlayer, #replayBtn, #autoplayNotice, .btn-replay-pulse.
     - Autoplay rejection catch handler (displays notice, activates replay pulse).
     - Autoplay success resolve handler (hides notice, removes replay pulse).
     - Replay click handler (resets currentTime=0, replays cleanly).
     - User gesture unlock mechanism (unlockAudioPlayback).
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


class TestAdversarialVITS36Sentences(unittest.TestCase):
    """Adversarial stress-testing of HmongVITS synthesis against acoustic, loudness, and packaging bounds."""

    @classmethod
    def setUpClass(cls):
        cls.tts = HmongVITS(sample_rate=16000, device="cpu")
        # 36 Diverse Hmong Daw RPA sentences spanning 1 syllable to 29 words
        cls.rpa_sentences = [
            # 1-8: Single Syllables covering all 8 White Hmong tones
            "zoo",                                          # 1. unmarked mid tone
            "ib",                                           # 2. -b tone (high)
            "kuv",                                          # 3. -v tone (mid-rising)
            "mus",                                          # 4. -s tone (low)
            "tsaug",                                        # 5. -g tone (breathy/creaky)
            "dej",                                          # 6. -j tone (high-falling)
            "tod",                                          # 7. -d tone (low-rising)
            "kaum",                                         # 8. -m tone (low-glottalized)
            # 9-10: 2 Syllables
            "nyob zoo",                                     # 9. standard greeting
            "ua tsaug",                                     # 10. standard thanks
            # 11-14: Hyphenated compound words
            "me-nyuam",                                     # 11. child
            "poj-niam",                                     # 12. woman/wife
            "txiv-neej",                                    # 13. man/husband
            "kwv-tij",                                      # 14. brothers/relatives
            # 15-19: Short daily phrases (3 to 6 words)
            "kuv nyob zoo",                                 # 15. I am well (3 words)
            "sib ntsib dua",                                # 16. See you again (3 words)
            "koj tuaj qhov twg tuaj",                       # 17. Where do you come from? (5 words)
            "koj lub npe hu li cas",                        # 18. What is your name? (6 words)
            "hnub no huab cua zoo heev",                    # 19. Today the weather is very nice (6 words)
            # 20-23: Medium daily sentences (8 to 10 words)
            "kuv xav mus tom khw yuav nqaij npuas",         # 20. I want to go to market to buy pork (8 words)
            "thov muab ib khob dej txias rau kuv",          # 21. Please give me a glass of cold water (8 words)
            "tsev kho mob nyob qhov twg thov pab qhia",      # 22. Where is hospital please help tell me (9 words)
            "kuv xav tau tshuaj mob taub hau thiab ua npaws", # 23. I need medicine for headache and fever (10 words)
            # 24-27: Complex compound sentences (11 to 16 words)
            "hnub no kuv yuav mus tom teb ua teb nrog kuv tsev neeg", # 24. (12 words)
            "tus kws kho mob hais tias koj yuav tsum tau noj tshuaj raws sij hawm", # 25. (14 words)
            "tag kis yog huab cua zoo peb yuav mus pem roob saib paj tawg zoo nkauj heev", # 26. (16 words)
            "ib ob peb plaub tsib rau xya yim cuaj kaum nees nkaum", # 27. (11 words number sequence)
            # 28-31: Long compound and multi-clause sentences (16 to 25 words)
            "kuv tsis paub hais lus hmoob zoo thov koj hais maj mam me ntsis rau kuv mloog", # 28. (16 words)
            "thaum ub muaj ib tug txiv ntxawm nyob rau hauv lub zos ntawd nws muaj ntau tus me nyuam heev", # 29. (18 words)
            "peb cov hmoob nyob rau toj siab muaj ntau yam kab lis kev cai zoo nkauj thiab qub txeeg qub teg heev uas tsim nyog ceev tseg", # 30. (23 words)
            "yog tias koj xav mus kawm ntawv nyob rau lub nroog loj koj yuav tsum mob siab kawm ntawv kom tiav thiab rov qab tuaj pab peb lub zos", # 31. (25 words)
            # 32-36: Punctuation, numerals, and narrative sentences
            "cais, txiav, nres! koj puas to taub zoo?",      # 32. punctuation clusters (7 words)
            "zeej, ib, ob, peb, plaub, tsib, rau, xya, yim, cuaj, kaum, nees nkaum, peb caug, plaub caug, tsib caug, rau caug, xya caug, yim caug, cuaj caug, ib puas.", # 33. numerals (21 words)
            "chaw kho mob, chaw muag tshuaj, chaw nres tsheb, chaw noj mov, chaw pw tsaug zog.", # 34. compound facilities (15 words)
            "qhov chaw no zoo nkauj heev, huab cua ntshiab, dej ntws txias, noog quaj nrov nrov rau saum ntoo.", # 35. descriptive (16 words)
            "tus tub hluas mus tua mos lwj pom ib tug tsov loj heev nyob hauv hav zoov tob heev nws ntshai kawg li thiaj li khiav ceev ceev rov los tsev." # 36. narrative (29 words)
        ]

    def test_01_verify_all_36_sentences_rms_dc_and_peak(self):
        """Stress-test: Verify all 36 RPA sentences strictly meet RMS [-20, -16] dBFS, peak <= 0.99, and DC < 1e-3."""
        self.assertGreaterEqual(len(self.rpa_sentences), 30, "Must test at least 30 varied RPA sentences")
        measurements: List[Dict[str, Any]] = []

        for idx, sentence in enumerate(self.rpa_sentences, 1):
            audio = self.tts.synthesize(sentence)
            self.assertIsInstance(audio, np.ndarray, f"Sentence #{idx} did not return ndarray")
            self.assertEqual(audio.dtype, np.float32, f"Sentence #{idx} dtype is not float32")
            self.assertGreater(len(audio), 0, f"Sentence #{idx} produced 0 samples: '{sentence}'")

            # Metrics
            dc = float(np.mean(audio))
            peak = float(np.max(np.abs(audio)))
            rms = float(np.sqrt(np.mean(audio ** 2)))
            dbfs = 20.0 * np.log10(rms) if rms > 0 else -999.0

            measurements.append({
                "index": idx,
                "text": sentence,
                "samples": len(audio),
                "duration_s": len(audio) / 16000.0,
                "dbfs": dbfs,
                "peak": peak,
                "dc": dc
            })

            with self.subTest(idx=idx, sentence=sentence[:30], samples=len(audio)):
                self.assertGreaterEqual(
                    dbfs, -20.0,
                    f"Sentence #{idx} '{sentence}' too quiet: {dbfs:.2f} dBFS < -20.0 dBFS"
                )
                self.assertLessEqual(
                    dbfs, -16.0,
                    f"Sentence #{idx} '{sentence}' too loud: {dbfs:.2f} dBFS > -16.0 dBFS"
                )
                self.assertLessEqual(
                    peak, 0.99,
                    f"Sentence #{idx} '{sentence}' exceeded ceiling: peak {peak:.4f} > 0.99"
                )
                self.assertLess(
                    abs(dc), 1e-3,
                    f"Sentence #{idx} '{sentence}' excessive DC offset: {dc:.6f} >= 1e-3"
                )

        # Print summary statistics
        all_dbfs = [m["dbfs"] for m in measurements]
        all_peaks = [m["peak"] for m in measurements]
        all_dcs = [abs(m["dc"]) for m in measurements]
        print(f"\n[Test 01 Summary: 36 Sentences Tested]")
        print(f"  RMS Range: [{min(all_dbfs):.2f}, {max(all_dbfs):.2f}] dBFS (Target: [-20.0, -16.0])")
        print(f"  Peak Range: [{min(all_peaks):.4f}, {max(all_peaks):.4f}] (Safety Ceiling: 0.99)")
        print(f"  Max |DC|: {max(all_dcs):.8f} (Max Allowed: 1e-3)")
        for m in measurements:
            status = "PASS" if -20.0 <= m["dbfs"] <= -16.0 and m["peak"] <= 0.99 and abs(m["dc"]) < 1e-3 else "FAIL"
            print(f"  [{status}] #{m['index']:02d}: dbfs={m['dbfs']:.2f}, peak={m['peak']:.4f}, len={m['samples']:6d} | '{m['text'][:35]}'")

    def test_02_wav_riff_header_and_soundfile_decoding(self):
        """Stress-test: Verify all 36 synthesized WAV streams have valid 44-byte headers and decode without warnings."""
        for idx, sentence in enumerate(self.rpa_sentences, 1):
            wav_bytes = self.tts.synthesize_to_bytes(sentence, format="wav")
            self.assertIsInstance(wav_bytes, bytes)
            self.assertGreater(len(wav_bytes), 44, f"Sentence #{idx} WAV shorter than 44 bytes")

            # 1. Binary RIFF header validation
            self.assertEqual(wav_bytes[:4], b"RIFF", f"Sentence #{idx} missing RIFF magic")
            self.assertEqual(wav_bytes[8:12], b"WAVE", f"Sentence #{idx} missing WAVE magic")
            self.assertEqual(wav_bytes[12:16], b"fmt ", f"Sentence #{idx} missing fmt subchunk")

            # 2. Standard wave module validation
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1, f"Sentence #{idx} not mono")
                self.assertEqual(wf.getsampwidth(), 2, f"Sentence #{idx} not 16-bit")
                self.assertEqual(wf.getframerate(), 16000, f"Sentence #{idx} not 16000 Hz")
                self.assertGreater(wf.getnframes(), 0, f"Sentence #{idx} 0 frames")

            # 3. Soundfile decoding: must catch zero warnings and verify PCM16 metrics
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                data, sr = sf.read(io.BytesIO(wav_bytes))

                self.assertEqual(len(caught_warnings), 0, f"Sentence #{idx} produced soundfile warnings: {caught_warnings}")
                self.assertEqual(sr, 16000)
                self.assertEqual(data.ndim, 1)
                self.assertGreater(len(data), 0)

                # Loudness and sanity on decoded soundfile data
                data_rms = float(np.sqrt(np.mean(data ** 2)))
                data_dbfs = 20.0 * np.log10(data_rms) if data_rms > 0 else -999.0
                data_peak = float(np.max(np.abs(data)))
                data_dc = float(np.mean(data))

                with self.subTest(idx=idx, sentence=sentence[:30]):
                    self.assertGreaterEqual(data_dbfs, -20.0)
                    self.assertLessEqual(data_dbfs, -16.0)
                    self.assertLessEqual(data_peak, 0.99)
                    self.assertLess(abs(data_dc), 1e-3)

    def test_03_speed_perturbation_acoustic_bounds(self):
        """Stress-test: Verify speed scaling (0.5x, 1.0x, 1.5x, 2.5x) preserves RMS corridor and zero clipping."""
        test_speeds = [0.5, 1.0, 1.5, 2.5]
        phrases = ["zoo", "nyob zoo", "hnub no huab cua zoo heev", "kuv xav mus tom khw yuav nqaij npuas"]

        for phrase in phrases:
            prev_samples = None
            for speed in test_speeds:
                audio = self.tts.synthesize(phrase, speed=speed)
                rms = float(np.sqrt(np.mean(audio ** 2)))
                dbfs = 20.0 * np.log10(rms) if rms > 0 else -999.0
                peak = float(np.max(np.abs(audio)))
                dc = float(np.mean(audio))

                with self.subTest(phrase=phrase, speed=speed):
                    # Audio constraints must hold under any playback speed
                    self.assertGreaterEqual(dbfs, -20.0, f"Speed {speed}x on '{phrase}' too quiet: {dbfs:.2f} dBFS")
                    self.assertLessEqual(dbfs, -16.0, f"Speed {speed}x on '{phrase}' too loud: {dbfs:.2f} dBFS")
                    self.assertLessEqual(peak, 0.99, f"Speed {speed}x on '{phrase}' clipped: {peak:.4f}")
                    self.assertLess(abs(dc), 1e-3, f"Speed {speed}x on '{phrase}' DC offset: {dc:.6f}")

                # Higher speed should result in shorter or equal sample duration
                if prev_samples is not None:
                    self.assertLessEqual(
                        len(audio), prev_samples,
                        f"Higher speed {speed} produced more samples than lower speed"
                    )
                prev_samples = len(audio)


class TestAdversarialDualInputAPIStress(unittest.TestCase):
    """Adversarial stress-testing of FastAPI Dual Input endpoints (/api/v1/t2s, /api/v1/s2s) under concurrency."""

    @classmethod
    def setUpClass(cls):
        cls.engine = get_engine(device="cpu", use_amp=False)
        app.state.engine = cls.engine
        cls.client = TestClient(app)

        cls.wav_file_path = os.path.join(PROJECT_ROOT, "tests", "data", "sample1_greeting.wav")
        cls.mp3_file_path = os.path.join(PROJECT_ROOT, "tests", "data", "sample1_greeting.mp3")
        cls.yt_file_path = os.path.join(PROJECT_ROOT, "tests", "data", "yt_speech", "yt_sample_01.wav")

    def test_01_rapid_concurrent_text_payloads_t2s(self):
        """Stress-test: Send 15 concurrent rapid text requests to POST /api/v1/t2s."""
        vi_sentences = [
            "xin chào bạn",
            "bác sĩ cho tôi hỏi bệnh viện ở đâu",
            "tôi cần mua thuốc cảm cúm và hạ sốt",
            "hôm nay thời tiết thế nào",
            "cho tôi một cốc nước ấm",
            "giá tiền này là bao nhiêu",
            "bây giờ là mấy giờ rồi",
            "tôi bị đau bụng dữ dội",
            "đi thẳng đường này rồi rẽ trái",
            "cảm ơn bạn rất nhiều",
            "hẹn gặp lại ngày mai",
            "chúc bạn một ngày tốt lành",
            "tôi muốn ăn phở bò",
            "nhà bạn có gần đây không",
            "rất vui được làm quen với bạn",
        ]

        def post_t2s(sentence: str):
            res = self.client.post("/api/v1/t2s", json={"text": sentence, "speed": 1.0})
            return sentence, res.status_code, res.headers, res.content

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(post_t2s, s) for s in vi_sentences]
            for fut in as_completed(futures):
                sent, status_code, headers, content = fut.result()
                self.assertEqual(status_code, 200, f"Failed on sentence: '{sent}'")
                self.assertEqual(headers.get("content-type"), "audio/wav")
                self.assertTrue(content.startswith(b"RIFF"))
                self.assertIn("x-vi-text", headers)
                self.assertIn("x-hmong-text", headers)
                self.assertGreater(len(headers["x-vi-text"]), 0)
                self.assertGreater(len(headers["x-hmong-text"]), 0)

                # Verify returned audio is valid WAV decodable by soundfile
                data, sr = sf.read(io.BytesIO(content))
                self.assertEqual(sr, 16000)
                self.assertGreater(len(data), 0)

    def test_02_rapid_concurrent_dual_input_s2s_text(self):
        """Stress-test: Send 10 concurrent requests to POST /api/v1/s2s with direct JSON & Form text."""
        test_cases = [
            ("json", {"text": "xin chào buổi sáng"}),
            ("form", {"text": "bệnh viện đa khoa ở đâu"}),
            ("json", {"text": "tôi cần mua bông băng thuốc đỏ"}),
            ("form", {"text": "trời hôm nay có mưa rào không"}),
            ("json", {"text": "cho tôi xin số điện thoại cấp cứu"}),
            ("form", {"text": "hãy giúp tôi với tôi bị ngã xe"}),
            ("json", {"text": "món ăn này bao nhiêu tiền một bát"}),
            ("form", {"text": "tôi muốn mua vé xe buýt đi chợ"}),
            ("json", {"text": "chúc bạn lên đường may mắn"}),
            ("form", {"text": "chúng tôi là người h'mông"}),
        ]

        def post_s2s(mode: str, payload: dict):
            if mode == "json":
                res = self.client.post("/api/v1/s2s", json=payload)
            else:
                res = self.client.post("/api/v1/s2s", data=payload)
            return mode, payload["text"], res.status_code, res.headers, res.content

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(post_s2s, m, p) for m, p in test_cases]
            for fut in as_completed(futures):
                mode, text, status_code, headers, content = fut.result()
                self.assertEqual(status_code, 200, f"Failed mode {mode} text '{text}'")
                self.assertEqual(headers.get("content-type"), "audio/wav")
                self.assertTrue(content.startswith(b"RIFF"))
                self.assertIn("x-vi-text", headers)
                self.assertIn("x-hmong-text", headers)

                data, sr = sf.read(io.BytesIO(content))
                self.assertEqual(sr, 16000)
                self.assertGreater(len(data), 0)

    def test_03_concurrent_audio_file_uploads_wav_and_mp3(self):
        """Stress-test: Post MP3 and WAV files concurrently to POST /api/v1/s2s."""
        self.assertTrue(os.path.exists(self.wav_file_path), "sample1_greeting.wav missing")
        self.assertTrue(os.path.exists(self.mp3_file_path), "sample1_greeting.mp3 missing")

        with open(self.wav_file_path, "rb") as f:
            wav_bytes = f.read()
        with open(self.mp3_file_path, "rb") as f:
            mp3_bytes = f.read()

        requests_to_run = [
            ("wav", "sample1_greeting.wav", wav_bytes, "audio/wav"),
            ("mp3", "sample1_greeting.mp3", mp3_bytes, "audio/mpeg"),
            ("wav", "sample1_greeting.wav", wav_bytes, "audio/wav"),
            ("mp3", "sample1_greeting.mp3", mp3_bytes, "audio/mpeg"),
        ]

        def post_file(fmt: str, fname: str, raw_bytes: bytes, ctype: str):
            res = self.client.post(
                "/api/v1/s2s",
                files={"file": (fname, raw_bytes, ctype)}
            )
            return fmt, res.status_code, res.headers, res.content

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(post_file, *r) for r in requests_to_run]
            for fut in as_completed(futures):
                fmt, status_code, headers, content = fut.result()
                self.assertEqual(status_code, 200, f"Failed audio format {fmt}")
                self.assertEqual(headers.get("content-type"), "audio/wav")
                self.assertTrue(content.startswith(b"RIFF"))
                self.assertIn("x-vi-text", headers)
                self.assertIn("x-hmong-text", headers)

                data, sr = sf.read(io.BytesIO(content))
                self.assertEqual(sr, 16000)
                self.assertGreater(len(data), 0)

    def test_04_mixed_concurrency_stress_all_modalities(self):
        """Stress-test: Concurrent mix of MP3, WAV, T2S text, and S2S text to verify zero race conditions or deadlocks."""
        with open(self.wav_file_path, "rb") as f:
            wav_bytes = f.read()
        with open(self.mp3_file_path, "rb") as f:
            mp3_bytes = f.read()

        tasks = [
            ("t2s", "hôm nay tôi rất khỏe", None),
            ("s2s_file_wav", "sample.wav", wav_bytes),
            ("s2s_text", "tôi muốn mua một cân táo", None),
            ("s2s_file_mp3", "sample.mp3", mp3_bytes),
            ("t2s", "bác sĩ hẹn tôi vào lúc chín giờ", None),
            ("s2s_file_wav", "sample.wav", wav_bytes),
        ]

        def run_task(task_type: str, arg1: str, arg2: bytes):
            if task_type == "t2s":
                res = self.client.post("/api/v1/t2s", json={"text": arg1})
            elif task_type == "s2s_text":
                res = self.client.post("/api/v1/s2s", json={"text": arg1})
            elif task_type == "s2s_file_wav":
                res = self.client.post("/api/v1/s2s", files={"file": (arg1, arg2, "audio/wav")})
            elif task_type == "s2s_file_mp3":
                res = self.client.post("/api/v1/s2s", files={"file": (arg1, arg2, "audio/mpeg")})
            else:
                raise ValueError(f"Unknown task {task_type}")
            return task_type, res.status_code, res.content

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_task, t, a1, a2) for t, a1, a2 in tasks]
            for fut in as_completed(futures):
                task_type, status_code, content = fut.result()
                self.assertEqual(status_code, 200, f"Failed task {task_type}")
                self.assertTrue(content.startswith(b"RIFF"))
                data, sr = sf.read(io.BytesIO(content))
                self.assertEqual(sr, 16000)
                self.assertGreater(len(data), 0)

    def test_05_boundary_and_malformed_payloads(self):
        """Stress-test: Empty, whitespace, special characters, unicode emoji, and oversized text."""
        # 1. Empty text to T2S
        res = self.client.post("/api/v1/t2s", json={"text": ""})
        self.assertIn(res.status_code, [400, 422])

        # 2. Blank whitespace text to T2S
        res = self.client.post("/api/v1/t2s", json={"text": "     \t\n  "})
        self.assertIn(res.status_code, [400, 422])

        # 3. Empty text to S2S
        res = self.client.post("/api/v1/s2s", json={"text": ""})
        self.assertEqual(res.status_code, 400)

        # 4. Special unicode / emojis in T2S: should handle gracefully without 500 error
        res = self.client.post("/api/v1/t2s", json={"text": "Xin chào 🌟 bác sĩ 🩺 123 !@#$%^&*()"})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.content.startswith(b"RIFF"))

        # 5. Long text payload (100+ words)
        long_text = "xin chào bác sĩ tôi bị đau bụng và sốt cao từ tối hôm qua đến nay " * 10
        res = self.client.post("/api/v1/t2s", json={"text": long_text})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.content.startswith(b"RIFF"))


class TestAdversarialHTML5PlayerDOM(unittest.TestCase):
    """Adversarial stress-testing of HTML5 player DOM and autoplay rejection mitigation logic."""

    @classmethod
    def setUpClass(cls):
        cls.html = INDEX_HTML

    def test_01_required_dom_elements_exist(self):
        """Verify presence of #nativeAudioPlayer, #replayBtn, #autoplayNotice, and #viTextInput."""
        self.assertIn('id="nativeAudioPlayer"', self.html, "Missing native audio player element")
        self.assertIn('<audio', self.html, "Missing <audio> tag")
        self.assertIn('controls', self.html, "Native audio player missing 'controls' attribute")
        self.assertIn('id="replayBtn"', self.html, "Missing replay button element")
        self.assertIn('id="autoplayNotice"', self.html, "Missing autoplay warning notice")
        self.assertIn('id="viTextInput"', self.html, "Missing Vietnamese text input textarea")
        self.assertIn('id="translateTextBtn"', self.html, "Missing text translate button")

    def test_02_autoplay_rejection_catch_handler(self):
        """Verify JavaScript handles browser autoplay rejection by showing notice and pulsing replayBtn."""
        # Must catch rejection on playPromise
        self.assertIn('audioPlayer.play()', self.html)
        self.assertIn('.catch(', self.html)
        # Catch handler must display notice and add pulse class
        self.assertIn('autoplayNotice.style.display = "block"', self.html)
        self.assertIn('replayBtn.classList.add("btn-replay-pulse")', self.html)
        # Success handler must hide notice and remove pulse
        self.assertIn('autoplayNotice.style.display = "none"', self.html)
        self.assertIn('replayBtn.classList.remove("btn-replay-pulse")', self.html)

    def test_03_replay_button_and_user_gesture_unlock(self):
        """Verify replayBtn click listener resets currentTime=0 and unlocks Web Audio playback."""
        self.assertIn('replayBtn.addEventListener("click"', self.html)
        self.assertIn('audioPlayer.currentTime = 0', self.html)
        self.assertIn('nativeAudioPlayer.currentTime = 0', self.html)
        self.assertIn('unlockAudioPlayback()', self.html)

    def test_04_pulse_css_animation_definition(self):
        """Verify .btn-replay-pulse CSS animation exists in styles to visually guide the user."""
        self.assertIn('.btn-replay-pulse', self.html)
        self.assertIn('keyframes', self.html)


if __name__ == "__main__":
    unittest.main()
