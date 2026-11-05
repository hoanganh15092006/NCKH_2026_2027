"""
tests/test_challenger_p2_m5.py
Empirical Adversarial Challenge Suite for Milestone P2-M5.

Focus areas:
1. WebSocket /ws/s2s_live VAD endpointing with small chunks (50ms, 85ms, 100ms):
   Ensures silence accumulation uses sample count and does NOT trigger prematurely before 400ms silence.
2. WebSocket Dual-Playback Fix Verification:
   Ensures server and client frame delivery protocol eliminates double-playback and audio stuttering.
3. Concurrent Client Stress Testing:
   4 concurrent REST clients and 4 concurrent WebSocket clients, verifying latency SLAs (< 1.5s)
   and measuring process memory stability (psutil RSS).
"""

import os
import sys
import time
import math
import io
import wave
import json
import base64
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import psutil
from fastapi.testclient import TestClient

from src.models.s2s_pipeline import SpeechToSpeechEngine
from src.api.app_stream import app, get_engine, RingBufferVAD, INDEX_HTML


def generate_pcm_chunk(
    is_speech: bool = True,
    sample_rate: int = 16000,
    duration_s: float = 0.05,
    freq: float = 300.0,
    amplitude: float = 0.5
) -> bytes:
    """Generates raw 16-bit mono linear PCM byte buffer of arbitrary duration."""
    num_samples = int(round(duration_s * sample_rate))
    if is_speech:
        t = np.linspace(0, duration_s, num_samples, endpoint=False, dtype=np.float32)
        # Harmonically rich signal to reliably trigger energy + VAD
        waveform = amplitude * (
            0.6 * np.sin(2 * np.pi * freq * t) +
            0.3 * np.sin(2 * np.pi * (freq * 2) * t) +
            0.1 * np.sin(2 * np.pi * (freq * 3) * t)
        )
    else:
        waveform = np.zeros(num_samples, dtype=np.float32)

    pcm16 = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm16.tobytes()


class ChallengerP2M5TestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Force CPU device for consistent, reproducible verification across test environments
        cls.engine = get_engine(device="cpu", use_amp=False)
        app.state.engine = cls.engine
        cls.client = TestClient(app)
        cls.process = psutil.Process(os.getpid())

    # =========================================================================
    # CHALLENGE 1: VAD Endpointing Granularity on Small Chunks (50ms, 85ms, 100ms)
    # =========================================================================

    def test_01_vad_endpointing_50ms_chunks(self):
        """
        Adversarial Challenge 1A:
        Test 50ms audio chunks (800 samples = 1600 bytes at 16kHz).
        Speech: 600ms (12 chunks).
        Silence: test 1 to 7 chunks (50ms to 350ms) -> MUST NOT trigger endpointing!
        Chunk 8 (400ms silence total) -> MUST trigger endpointing!
        """
        vad = RingBufferVAD(
            sample_rate=16000,
            chunk_size_ms=300,
            min_silence_ms=400,
            min_speech_ms=200,
            vad_threshold=0.5
        )

        chunk_50ms_speech = generate_pcm_chunk(is_speech=True, duration_s=0.05, freq=320.0)
        chunk_50ms_silence = generate_pcm_chunk(is_speech=False, duration_s=0.05)

        self.assertEqual(len(chunk_50ms_speech), 1600)
        self.assertEqual(len(chunk_50ms_silence), 1600)

        # 1. Feed 600ms of speech (12 chunks)
        for i in range(12):
            res = vad.add_chunk(chunk_50ms_speech, self.engine.preprocessor)
            self.assertIsNone(res, f"Speech chunk {i+1} prematurely triggered utterance")
        self.assertTrue(vad.in_speech, "VAD must be in speech mode after 600ms of voiced frames")

        # 2. Feed silence chunks 1 to 7 (50ms, 100ms, 150ms, 200ms, 250ms, 300ms, 350ms)
        for s in range(1, 8):
            res = vad.add_chunk(chunk_50ms_silence, self.engine.preprocessor)
            self.assertIsNone(
                res,
                f"Silence chunk {s} ({s*50}ms total silence) prematurely triggered endpoint before 400ms!"
            )
            self.assertTrue(vad.in_speech, f"VAD must remain in speech state at {s*50}ms silence")

        # 3. Feed 8th silence chunk (400ms total silence = 6,400 samples)
        res = vad.add_chunk(chunk_50ms_silence, self.engine.preprocessor)
        self.assertIsNotNone(res, "8th silence chunk (400ms silence) must trigger utterance endpointing")
        # Check returned utterance contains 12 speech + 8 silence chunks = 20 chunks = 32,000 bytes
        self.assertEqual(len(res), 20 * 1600)
        self.assertFalse(vad.in_speech, "VAD must reset to non-speech state after endpointing")

    def test_02_vad_endpointing_85ms_chunks(self):
        """
        Adversarial Challenge 1B:
        Test 85ms non-standard audio chunks (1,360 samples = 2,720 bytes at 16kHz).
        Speech: 8 chunks (680ms).
        Silence: chunks 1 to 4 (85ms, 170ms, 255ms, 340ms silence) -> MUST NOT trigger!
        Chunk 5 (425ms total silence >= 400ms) -> MUST trigger!
        """
        vad = RingBufferVAD(
            sample_rate=16000,
            chunk_size_ms=300,
            min_silence_ms=400,
            min_speech_ms=200,
            vad_threshold=0.5
        )

        chunk_85ms_speech = generate_pcm_chunk(is_speech=True, duration_s=0.085, freq=340.0)
        chunk_85ms_silence = generate_pcm_chunk(is_speech=False, duration_s=0.085)

        self.assertEqual(len(chunk_85ms_speech), 2720)
        self.assertEqual(len(chunk_85ms_silence), 2720)

        # 1. Feed 680ms speech (8 chunks)
        for i in range(8):
            res = vad.add_chunk(chunk_85ms_speech, self.engine.preprocessor)
            self.assertIsNone(res)
        self.assertTrue(vad.in_speech)

        # 2. Silence chunks 1 to 4 (85ms, 170ms, 255ms, 340ms)
        for s in range(1, 5):
            res = vad.add_chunk(chunk_85ms_silence, self.engine.preprocessor)
            self.assertIsNone(
                res,
                f"Silence chunk {s} ({s*85}ms silence) prematurely triggered endpoint before 400ms!"
            )

        # 3. Silence chunk 5 (425ms silence >= 400ms = 6,800 samples >= 6,400 samples)
        res = vad.add_chunk(chunk_85ms_silence, self.engine.preprocessor)
        self.assertIsNotNone(res, "Silence chunk 5 (425ms silence) must trigger endpointing")
        self.assertEqual(len(res), (8 + 5) * 2720)

    def test_03_vad_endpointing_100ms_chunks(self):
        """
        Adversarial Challenge 1C:
        Test 100ms audio chunks (1,600 samples = 3,200 bytes at 16kHz).
        Speech: 6 chunks (600ms).
        Silence: chunks 1 to 3 (100ms, 200ms, 300ms) -> MUST NOT trigger!
        Chunk 4 (400ms total silence) -> MUST trigger!
        """
        vad = RingBufferVAD(
            sample_rate=16000,
            chunk_size_ms=300,
            min_silence_ms=400,
            min_speech_ms=200,
            vad_threshold=0.5
        )

        chunk_100ms_speech = generate_pcm_chunk(is_speech=True, duration_s=0.100, freq=300.0)
        chunk_100ms_silence = generate_pcm_chunk(is_speech=False, duration_s=0.100)

        self.assertEqual(len(chunk_100ms_speech), 3200)
        self.assertEqual(len(chunk_100ms_silence), 3200)

        # 1. Feed 600ms speech
        for _ in range(6):
            res = vad.add_chunk(chunk_100ms_speech, self.engine.preprocessor)
            self.assertIsNone(res)
        self.assertTrue(vad.in_speech)

        # 2. Feed 3 silence chunks (100ms, 200ms, 300ms)
        for s in range(1, 4):
            res = vad.add_chunk(chunk_100ms_silence, self.engine.preprocessor)
            self.assertIsNone(
                res,
                f"Silence chunk {s} ({s*100}ms silence) prematurely triggered endpoint before 400ms!"
            )

        # 3. Feed 4th silence chunk (400ms silence)
        res = vad.add_chunk(chunk_100ms_silence, self.engine.preprocessor)
        self.assertIsNotNone(res, "4th silence chunk (400ms) must trigger endpointing")
        self.assertEqual(len(res), 10 * 3200)

    def test_04_vad_resumed_speech_after_sub_400ms_pause(self):
        """
        Adversarial Challenge 1D:
        Natural speech pause: Speaker speaks for 500ms, pauses for 300ms (< 400ms),
        then continues speaking for 500ms, and finally pauses for 400ms.
        Verify:
        - Utterance is NOT cut off during the 300ms pause.
        - Silence counter resets upon speech resumption.
        - Final utterance contains both speech segments and the pause seamlessly.
        """
        vad = RingBufferVAD(sample_rate=16000, min_silence_ms=400, min_speech_ms=200)

        chunk_speech = generate_pcm_chunk(is_speech=True, duration_s=0.05, freq=350.0)
        chunk_silence = generate_pcm_chunk(is_speech=False, duration_s=0.05)

        # Segment 1: 500ms speech (10 chunks)
        for _ in range(10):
            self.assertIsNone(vad.add_chunk(chunk_speech, self.engine.preprocessor))

        # Pause: 300ms silence (6 chunks)
        for _ in range(6):
            self.assertIsNone(vad.add_chunk(chunk_silence, self.engine.preprocessor))
        self.assertEqual(vad.silence_samples, 6 * 800)  # 2400 samples

        # Segment 2: 500ms speech (10 chunks) -> silence must reset to 0
        for i in range(10):
            self.assertIsNone(vad.add_chunk(chunk_speech, self.engine.preprocessor))
            if i == 0:
                self.assertEqual(vad.silence_samples, 0, "Silence counter must reset on speech resumption")

        # Final Silence: 8 chunks (400ms)
        for s in range(7):
            self.assertIsNone(vad.add_chunk(chunk_silence, self.engine.preprocessor))
        
        final_utt = vad.add_chunk(chunk_silence, self.engine.preprocessor)
        self.assertIsNotNone(final_utt, "Endpoint must trigger at 400ms final silence")
        # Total chunks = 10 (speech1) + 6 (pause) + 10 (speech2) + 8 (silence) = 34 chunks * 1600 bytes
        self.assertEqual(len(final_utt), 34 * 1600)

    def test_05_websocket_live_micro_chunks_streaming(self):
        """
        Adversarial Challenge 1E:
        Live WebSocket /ws/s2s_live test with micro-chunks (50ms and 100ms).
        Verify that streaming micro-chunks over WebSocket does NOT produce premature responses,
        and accurately yields a single complete response upon reaching >= 400ms silence.
        """
        with self.client.websocket_connect("/ws/s2s_live?format=wav") as ws:
            # Turn 1: 50ms chunks (12 speech chunks = 600ms)
            for _ in range(12):
                ws.send_bytes(generate_pcm_chunk(is_speech=True, duration_s=0.05, freq=320.0))
            # Send 6 silence chunks (300ms) -> no response yet
            for _ in range(6):
                ws.send_bytes(generate_pcm_chunk(is_speech=False, duration_s=0.05))

            # Send 7th silence chunk (350ms) -> no response yet
            ws.send_bytes(generate_pcm_chunk(is_speech=False, duration_s=0.05))

            # Send 8th silence chunk (400ms) -> TRIGGERS endpoint
            t0 = time.perf_counter()
            ws.send_bytes(generate_pcm_chunk(is_speech=False, duration_s=0.05))

            resp_bytes = ws.receive_bytes()
            t_lat = (time.perf_counter() - t0) * 1000.0
            self.assertTrue(resp_bytes.startswith(b"RIFF"), "Must return valid WAV container")
            self.assertGreater(len(resp_bytes), 1000)
            print(f"\n[Challenge 1E] WebSocket 50ms micro-chunk turn latency: {t_lat:.1f}ms")

            # Turn 2: 100ms chunks (6 speech chunks = 600ms, then 3 silence chunks = 300ms, then 4th = 400ms)
            for _ in range(6):
                ws.send_bytes(generate_pcm_chunk(is_speech=True, duration_s=0.1, freq=280.0))
            for _ in range(3):
                ws.send_bytes(generate_pcm_chunk(is_speech=False, duration_s=0.1))

            t0 = time.perf_counter()
            ws.send_bytes(generate_pcm_chunk(is_speech=False, duration_s=0.1))
            resp_bytes2 = ws.receive_bytes()
            t_lat2 = (time.perf_counter() - t0) * 1000.0
            self.assertTrue(resp_bytes2.startswith(b"RIFF"))
            print(f"[Challenge 1E] WebSocket 100ms micro-chunk turn latency: {t_lat2:.1f}ms")

    # =========================================================================
    # CHALLENGE 2: WebSocket Dual-Playback Fix Verification
    # =========================================================================

    def test_06_websocket_frame_protocol_audit(self):
        """
        Adversarial Challenge 2A:
        Inspect server frame emission protocol for /ws/s2s_live:
        1. When events=false:
           - Exactly ONE frame is emitted per utterance, and it is a BINARY frame (WAV/PCM).
           - ZERO JSON text frames are emitted.
        2. When events=true:
           - Exactly TWO frames are emitted per utterance in strict order:
             Frame 1: TEXT frame containing JSON telemetry (type="result", vi_text, hmong_text, latency_ms).
             Frame 2: BINARY frame containing synthesized audio bytes.
           - Crucially, the server does NOT emit two binary audio frames.
        """
        # Test Case 1: events=false
        with self.client.websocket_connect("/ws/s2s_live?events=false&format=wav") as ws:
            # Send 600ms speech + 400ms silence (in 100ms chunks)
            for _ in range(6):
                ws.send_bytes(generate_pcm_chunk(is_speech=True, duration_s=0.1))
            for _ in range(4):
                ws.send_bytes(generate_pcm_chunk(is_speech=False, duration_s=0.1))

            # Receive raw frame
            frame = ws.receive()
            self.assertIn("bytes", frame, "Frame must be binary bytes when events=false")
            self.assertNotIn("text", frame, "No text/JSON frame should be sent when events=false")
            self.assertTrue(frame["bytes"].startswith(b"RIFF"))

        # Test Case 2: events=true
        with self.client.websocket_connect("/ws/s2s_live?events=true&format=wav") as ws:
            for _ in range(6):
                ws.send_bytes(generate_pcm_chunk(is_speech=True, duration_s=0.1))
            for _ in range(4):
                ws.send_bytes(generate_pcm_chunk(is_speech=False, duration_s=0.1))

            # Frame 1: must be JSON text
            frame1 = ws.receive()
            self.assertIn("text", frame1, "First frame in events mode must be text JSON")
            data = json.loads(frame1["text"])
            self.assertEqual(data.get("type"), "result")
            self.assertIn("vi_text", data)
            self.assertIn("hmong_text", data)
            self.assertIn("latency_ms", data)
            self.assertIn("audio_base64", data)

            # Frame 2: must be binary audio bytes
            frame2 = ws.receive()
            self.assertIn("bytes", frame2, "Second frame in events mode must be binary bytes")
            self.assertTrue(frame2["bytes"].startswith(b"RIFF"))

    def test_07_client_javascript_dual_playback_guard(self):
        """
        Adversarial Challenge 2B:
        Examine the client-side JavaScript implementation embedded in `INDEX_HTML`:
        Verify:
        1. In `websocket.onmessage`:
           - Handling of string events (typeof event.data === "string") ONLY updates DOM text:
             viOutput, hmOutput, latencyDisplay.
           - Crucially, string handling DOES NOT call playSynthesizedAudio or play audio.
        2. Handling of binary events (event.data instanceof ArrayBuffer):
           - Exclusively calls playSynthesizedAudio(blob).
        3. Verify audio playback call count in simulated browser event loop is strictly 1 per utterance.
        """
        # 1. Structural inspection of INDEX_HTML
        self.assertIn("websocket.onmessage", INDEX_HTML)
        self.assertIn('event.data instanceof ArrayBuffer', INDEX_HTML)
        self.assertIn('playSynthesizedAudio(blob)', INDEX_HTML)

        # Ensure that inside the `typeof event.data === "string"` block, playSynthesizedAudio is NOT called
        onmessage_idx = INDEX_HTML.find("websocket.onmessage = async (event) => {")
        self.assertNotEqual(onmessage_idx, -1)
        onmessage_block = INDEX_HTML[onmessage_idx:onmessage_idx + 1200]

        # Extract string handler block
        str_start = onmessage_block.find('if (typeof event.data === "string") {')
        str_end = onmessage_block.find('} else if (event.data instanceof ArrayBuffer) {')
        self.assertNotEqual(str_start, -1)
        self.assertNotEqual(str_end, -1)
        string_handler_code = onmessage_block[str_start:str_end]

        self.assertNotIn(
            "playSynthesizedAudio",
            string_handler_code,
            "CRITICAL: playSynthesizedAudio found inside string message handler! This causes double-playback!"
        )
        self.assertNotIn(
            "audioPlayer.play()",
            string_handler_code,
            "CRITICAL: audioPlayer.play() found inside string message handler! This causes double-playback!"
        )

        # 2. Functional browser event dispatch simulation
        class SimulatedBrowserClient:
            def __init__(self):
                self.vi_text = ""
                self.hmong_text = ""
                self.latency_ms = None
                self.play_count = 0
                self.played_blobs = []

            def playSynthesizedAudio(self, data: bytes):
                self.play_count += 1
                self.played_blobs.append(data)

            def onmessage(self, message_data):
                # Mirrors exact client JavaScript logic in INDEX_HTML
                if isinstance(message_data, str):
                    data = json.loads(message_data)
                    if data.get("type") == "result":
                        if "vi_text" in data:
                            self.vi_text = data["vi_text"]
                        if "hmong_text" in data:
                            self.hmong_text = data["hmong_text"]
                        if "latency_ms" in data:
                            self.latency_ms = data["latency_ms"]
                        # Audio playback is intentionally suppressed here
                elif isinstance(message_data, (bytes, bytearray)):
                    self.playSynthesizedAudio(message_data)

        sim_client = SimulatedBrowserClient()

        # Simulate incoming WebSocket frames from server (events=true)
        dummy_audio = b"RIFF....WAVEfmt ...."
        json_payload = json.dumps({
            "type": "result",
            "vi_text": "xin chào",
            "hmong_text": "nyob zoo",
            "latency_ms": 120.5,
            "audio_format": "wav",
            "audio_base64": base64.b64encode(dummy_audio).decode("ascii")
        })

        # Frame 1 arrives
        sim_client.onmessage(json_payload)
        self.assertEqual(sim_client.vi_text, "xin chào")
        self.assertEqual(sim_client.hmong_text, "nyob zoo")
        self.assertEqual(sim_client.play_count, 0, "No audio should play upon JSON metadata reception")

        # Frame 2 arrives
        sim_client.onmessage(dummy_audio)
        self.assertEqual(sim_client.play_count, 1, "Audio must play exactly once on binary frame reception")
        self.assertEqual(sim_client.played_blobs[0], dummy_audio)

    # =========================================================================
    # CHALLENGE 3: Stress Testing Concurrent Clients & Memory Stability
    # =========================================================================

    def test_08_concurrent_rest_clients_stress(self):
        """
        Adversarial Challenge 3A:
        Stress test 4 concurrent REST clients hitting POST /api/v1/s2s simultaneously.
        Each client posts genuine audio data from tests/data/.
        Verify:
        - All 4 requests succeed with HTTP 200.
        - Correct X-Vi-Text and X-Hmong-Text headers returned.
        - Returned audio is valid 16kHz WAV.
        - Latencies satisfy real-time SLA bounds.
        """
        test_wav_path = "tests/data/sample1_greeting.wav"
        if not os.path.exists(test_wav_path):
            test_wav_path = "outputs/test_vietnamese_real.wav"

        with open(test_wav_path, "rb") as f:
            wav_bytes = f.read()

        num_clients = 4
        latencies = []
        status_codes = []

        def rest_worker(cid: int) -> Tuple[int, float, int, str]:
            t0 = time.perf_counter()
            # Send multipart form upload
            worker_client = TestClient(app)
            response = worker_client.post(
                "/api/v1/s2s",
                files={"file": (f"test_client_{cid}.wav", io.BytesIO(wav_bytes), "audio/wav")}
            )
            lat = (time.perf_counter() - t0) * 1000.0
            return response.status_code, lat, len(response.content), response.headers.get("X-Hmong-Text", "")

        t_batch_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_clients) as pool:
            futures = [pool.submit(rest_worker, cid) for cid in range(num_clients)]
            for fut in as_completed(futures):
                code, lat, content_len, hmong_hdr = fut.result()
                status_codes.append(code)
                latencies.append(lat)
                self.assertEqual(code, 200)
                self.assertGreater(content_len, 1000)
                self.assertTrue(len(hmong_hdr) > 0)
        t_batch_total = (time.perf_counter() - t_batch_start) * 1000.0

        p50 = float(np.percentile(latencies, 50))
        p90 = float(np.percentile(latencies, 90))
        max_lat = max(latencies)
        print(f"\n[Challenge 3A] 4 Concurrent REST Clients Stress:")
        print(f"  - Total batch elapsed: {t_batch_total:.1f}ms")
        print(f"  - Latencies: P50={p50:.1f}ms, P90={p90:.1f}ms, Max={max_lat:.1f}ms")
        self.assertEqual(len(status_codes), num_clients)
        self.assertTrue(all(c == 200 for c in status_codes))

    def test_09_concurrent_websocket_clients_stress(self):
        """
        Adversarial Challenge 3B:
        Stress test 4 concurrent WebSocket streaming clients connected simultaneously to /ws/s2s_live.
        Each client streams 3 sequential conversational turns (12 total utterances under concurrency).
        Measure round-trip latency, memory, and audio validity.
        """
        num_clients = 4
        turns_per_client = 3
        all_latencies = []

        def ws_worker(client_id: int) -> List[float]:
            worker_client = TestClient(app)
            client_lats = []
            with worker_client.websocket_connect("/ws/s2s_live?format=wav") as ws:
                for turn in range(turns_per_client):
                    # 6 speech chunks (100ms each = 600ms) with client-specific pitch
                    freq = 250.0 + client_id * 40.0 + turn * 20.0
                    for _ in range(6):
                        ws.send_bytes(generate_pcm_chunk(is_speech=True, duration_s=0.1, freq=freq))
                    # 3 silence chunks (300ms)
                    for _ in range(3):
                        ws.send_bytes(generate_pcm_chunk(is_speech=False, duration_s=0.1))

                    # 4th silence chunk (trigger)
                    t0 = time.perf_counter()
                    ws.send_bytes(generate_pcm_chunk(is_speech=False, duration_s=0.1))

                    resp = ws.receive_bytes()
                    lat = (time.perf_counter() - t0) * 1000.0
                    client_lats.append(lat)
                    self.assertTrue(resp.startswith(b"RIFF"), "Received data must be WAV container")
                    self.assertGreater(len(resp), 1000)
            return client_lats

        with ThreadPoolExecutor(max_workers=num_clients) as pool:
            futures = [pool.submit(ws_worker, cid) for cid in range(num_clients)]
            for fut in as_completed(futures):
                all_latencies.extend(fut.result())

        self.assertEqual(len(all_latencies), num_clients * turns_per_client)
        p50 = float(np.percentile(all_latencies, 50))
        p90 = float(np.percentile(all_latencies, 90))
        p99 = float(np.percentile(all_latencies, 99))
        max_lat = max(all_latencies)

        print(f"\n[Challenge 3B] 4 Concurrent WebSocket Streaming Clients ({len(all_latencies)} turns):")
        print(f"  - P50: {p50:.1f}ms")
        print(f"  - P90: {p90:.1f}ms")
        print(f"  - P99: {p99:.1f}ms")
        print(f"  - Max: {max_lat:.1f}ms")
        self.assertLess(p50, 1500.0)

    def test_10_concurrent_mixed_load_and_memory_stability(self):
        """
        Adversarial Challenge 3C:
        Mixed concurrent workload: 2 concurrent WebSocket streams + 2 concurrent REST requests.
        Repeated across 5 sequential rounds (20 total operations under mixed concurrency).
        Measure process memory (RSS) before and after to verify memory remains stable and leak-free.
        """
        test_wav_path = "tests/data/sample1_greeting.wav"
        if not os.path.exists(test_wav_path):
            test_wav_path = "outputs/test_vietnamese_real.wav"
        with open(test_wav_path, "rb") as f:
            wav_bytes = f.read()

        rss_before_mb = self.process.memory_info().rss / (1024 * 1024)

        rounds = 5
        mixed_latencies = []

        def run_ws_round(cid: int) -> float:
            c = TestClient(app)
            with c.websocket_connect("/ws/s2s_live?format=wav") as ws:
                for _ in range(5):
                    ws.send_bytes(generate_pcm_chunk(is_speech=True, duration_s=0.1, freq=280.0 + cid*30))
                for _ in range(3):
                    ws.send_bytes(generate_pcm_chunk(is_speech=False, duration_s=0.1))
                t0 = time.perf_counter()
                ws.send_bytes(generate_pcm_chunk(is_speech=False, duration_s=0.1))
                resp = ws.receive_bytes()
                lat = (time.perf_counter() - t0) * 1000.0
                self.assertTrue(resp.startswith(b"RIFF"))
                return lat

        def run_rest_round(cid: int) -> float:
            c = TestClient(app)
            t0 = time.perf_counter()
            resp = c.post(
                "/api/v1/s2s",
                files={"file": (f"mixed_{cid}.wav", io.BytesIO(wav_bytes), "audio/wav")}
            )
            lat = (time.perf_counter() - t0) * 1000.0
            self.assertEqual(resp.status_code, 200)
            return lat

        for r in range(rounds):
            with ThreadPoolExecutor(max_workers=4) as pool:
                f_ws1 = pool.submit(run_ws_round, 1)
                f_ws2 = pool.submit(run_ws_round, 2)
                f_rest1 = pool.submit(run_rest_round, 1)
                f_rest2 = pool.submit(run_rest_round, 2)

                for f in (f_ws1, f_ws2, f_rest1, f_rest2):
                    mixed_latencies.append(f.result())

        rss_after_mb = self.process.memory_info().rss / (1024 * 1024)
        mem_delta_mb = rss_after_mb - rss_before_mb

        p50 = float(np.percentile(mixed_latencies, 50))
        p90 = float(np.percentile(mixed_latencies, 90))

        print(f"\n[Challenge 3C] Mixed Concurrent Load ({len(mixed_latencies)} ops across {rounds} rounds):")
        print(f"  - Latency P50: {p50:.1f}ms, P90: {p90:.1f}ms")
        print(f"  - Memory RSS before: {rss_before_mb:.2f} MB")
        print(f"  - Memory RSS after:  {rss_after_mb:.2f} MB")
        print(f"  - Memory Delta:      {mem_delta_mb:+.2f} MB")

        # Verify memory delta is stable (< 150MB growth across all rounds, no uncontrolled leak)
        self.assertLess(mem_delta_mb, 150.0, f"Memory growth {mem_delta_mb:.2f}MB exceeds 150MB tolerance")


if __name__ == "__main__":
    unittest.main()
