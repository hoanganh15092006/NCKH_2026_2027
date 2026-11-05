"""
tests/test_empirical_streaming_challenge.py
Empirical Streaming Latency, Concurrency, and WebSocket Challenge Suite.

Implements rigorous empirical challenges:
1. Benchmark SpeechToSpeechEngine.process_stream under high-frequency chunk feeds (bursts, micro-chunks, jitter).
2. Benchmark WebSocket /ws/s2s_live with streaming PCM chunks across single and concurrent connections.
3. Compute and verify empirical P50, P90, P99 end-to-end latencies strictly < 1,500ms.
4. Benchmark HmongVITS TTS Real-Time Factor (RTF) across diverse RPA inputs and verify RTF < 1.0.
"""

import time
import math
import io
import wave
import json
import random
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from fastapi.testclient import TestClient

from src.models.s2s_pipeline import SpeechToSpeechEngine
from src.models.tts_module import HmongVITS
from src.api.app_stream import app, get_engine


def make_pcm_chunk(
    is_speech: bool = True,
    sample_rate: int = 16000,
    duration_s: float = 0.3,
    freq: float = 350.0,
    amplitude: float = 0.5
) -> bytes:
    """Creates a raw 16-bit linear PCM byte buffer."""
    num_samples = int(duration_s * sample_rate)
    if is_speech:
        t = np.linspace(0, duration_s, num_samples, endpoint=False, dtype=np.float32)
        # Add fundamental + harmonics for realistic speech-like energy
        waveform = amplitude * (
            0.6 * np.sin(2 * np.pi * freq * t) +
            0.3 * np.sin(2 * np.pi * (freq * 2) * t) +
            0.1 * np.sin(2 * np.pi * (freq * 3) * t)
        )
    else:
        waveform = np.zeros(num_samples, dtype=np.float32)

    pcm16 = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm16.tobytes()


class EmpiricalStreamingChallenge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Force CPU device for consistent, reproducible benchmarking on host
        cls.engine = get_engine(device="cpu", use_amp=False)
        app.state.engine = cls.engine
        cls.client = TestClient(app)
        cls.tts = cls.engine.tts

    # -----------------------------------------------------------------------
    # Challenge 1: Benchmark SpeechToSpeechEngine.process_stream under high-frequency chunk feeds
    # -----------------------------------------------------------------------
    def test_01_process_stream_standard_burst_feed(self):
        """
        Feed 300ms chunks back-to-back with zero delay across 10 sequential utterances.
        Verify that process_stream correctly handles high-frequency burst feeds,
        segments utterances via VAD, and produces valid audio bytes for each utterance.
        """
        latencies: List[float] = []
        outputs: List[bytes] = []

        num_utterances = 10
        chunks_per_speech = 3  # 900ms speech
        chunks_per_silence = 2  # 600ms silence hangover

        def burst_generator():
            for u in range(num_utterances):
                # Voiced chunks
                for _ in range(chunks_per_speech):
                    yield make_pcm_chunk(is_speech=True, duration_s=0.3)
                # Silence chunks triggering endpoint
                for _ in range(chunks_per_silence):
                    yield make_pcm_chunk(is_speech=False, duration_s=0.3)

        t_start = time.perf_counter()
        stream_iter = self.engine.process_stream(burst_generator())
        
        last_t = t_start
        for pcm_bytes in stream_iter:
            now = time.perf_counter()
            latencies.append((now - last_t) * 1000.0)
            last_t = now
            outputs.append(pcm_bytes)

        self.assertEqual(len(outputs), num_utterances, f"Expected {num_utterances} outputs, got {len(outputs)}")
        for out in outputs:
            self.assertGreater(len(out), 0, "Output PCM bytes must not be empty")
            # Verify valid 16-bit PCM length (even number of bytes)
            self.assertEqual(len(out) % 2, 0, "PCM byte length must be multiple of 2 (16-bit)")

        print(f"\n[Test 1.1] Standard Burst Feed (10 utterances): Mean interval = {np.mean(latencies):.1f}ms, P50 = {np.median(latencies):.1f}ms")

    def test_02_process_stream_micro_chunks_high_frequency(self):
        """
        Feed 100ms micro-chunks (1,600 samples, 3,200 bytes) at high frequency.
        Tests VAD aggregation across 9 speech micro-chunks + 6 silence micro-chunks.
        """
        latencies: List[float] = []
        outputs: List[bytes] = []

        num_utterances = 8
        micro_speech_chunks = 9   # 900ms total
        micro_silence_chunks = 6  # 600ms total

        def micro_chunk_generator():
            for _ in range(num_utterances):
                for _ in range(micro_speech_chunks):
                    yield make_pcm_chunk(is_speech=True, duration_s=0.1)
                for _ in range(micro_silence_chunks):
                    yield make_pcm_chunk(is_speech=False, duration_s=0.1)

        t_start = time.perf_counter()
        stream_iter = self.engine.process_stream(micro_chunk_generator())
        last_t = t_start

        for pcm_bytes in stream_iter:
            now = time.perf_counter()
            latencies.append((now - last_t) * 1000.0)
            last_t = now
            outputs.append(pcm_bytes)

        self.assertEqual(len(outputs), num_utterances)
        for out in outputs:
            self.assertGreater(len(out), 0)

        print(f"[Test 1.2] Micro-Chunk Feed (8 utterances): Mean interval = {np.mean(latencies):.1f}ms, P50 = {np.median(latencies):.1f}ms")

    def test_03_process_stream_jittered_feeds(self):
        """
        Stress feed with simulated network arrival jitter (random 0 to 5ms sleep between chunks).
        Verify that latency and audio integrity remain intact under network jitter.
        """
        latencies: List[float] = []
        outputs: List[bytes] = []
        num_utterances = 6

        def jittered_generator():
            for _ in range(num_utterances):
                for _ in range(3):
                    time.sleep(random.uniform(0.001, 0.005))
                    yield make_pcm_chunk(is_speech=True, duration_s=0.3)
                for _ in range(2):
                    time.sleep(random.uniform(0.001, 0.005))
                    yield make_pcm_chunk(is_speech=False, duration_s=0.3)

        t_start = time.perf_counter()
        for pcm_bytes in self.engine.process_stream(jittered_generator()):
            now = time.perf_counter()
            latencies.append((now - t_start) * 1000.0)
            t_start = now
            outputs.append(pcm_bytes)

        self.assertEqual(len(outputs), num_utterances)
        print(f"[Test 1.3] Jittered Feed (6 utterances): Mean interval = {np.mean(latencies):.1f}ms, P50 = {np.median(latencies):.1f}ms")

    def test_04_process_stream_events_latency_metrics(self):
        """
        Benchmark process_stream_events for fine-grained internal pipeline latency tracking.
        """
        num_utterances = 10
        def stream_gen():
            for _ in range(num_utterances):
                for _ in range(3):
                    yield make_pcm_chunk(is_speech=True, duration_s=0.3)
                for _ in range(2):
                    yield make_pcm_chunk(is_speech=False, duration_s=0.3)

        events = list(self.engine.process_stream_events(stream_gen()))
        self.assertEqual(len(events), num_utterances)

        internal_latencies = [ev["latency_ms"] for ev in events]
        p50 = float(np.percentile(internal_latencies, 50))
        p90 = float(np.percentile(internal_latencies, 90))
        p99 = float(np.percentile(internal_latencies, 99))

        print(f"[Test 1.4] process_stream_events Latencies: P50={p50:.1f}ms, P90={p90:.1f}ms, P99={p99:.1f}ms, Max={max(internal_latencies):.1f}ms")
        self.assertLess(p50, 1500.0)
        self.assertLess(p90, 1500.0)
        self.assertLess(p99, 1500.0)

    # -----------------------------------------------------------------------
    # Challenge 2: Benchmark WebSocket /ws/s2s_live with streaming PCM chunks
    # -----------------------------------------------------------------------
    def test_05_websocket_single_client_sequential_streaming(self):
        """
        Benchmark WebSocket /ws/s2s_live with sequential streaming of 15 conversational utterances.
        Measure client-side round-trip latency from final silence chunk until binary response received.
        """
        latencies: List[float] = []
        num_turns = 15

        with self.client.websocket_connect("/ws/s2s_live?format=wav") as ws:
            for turn in range(num_turns):
                # Send 3 speech chunks (900ms)
                for _ in range(3):
                    ws.send_bytes(make_pcm_chunk(is_speech=True, duration_s=0.3))
                # Send 1st silence chunk
                ws.send_bytes(make_pcm_chunk(is_speech=False, duration_s=0.3))

                # Send 2nd silence chunk (triggers VAD endpoint) & record timestamp
                t_trigger = time.perf_counter()
                ws.send_bytes(make_pcm_chunk(is_speech=False, duration_s=0.3))

                # Await synthesized audio response
                response_bytes = ws.receive_bytes()
                t_recv = time.perf_counter()
                rt_latency_ms = (t_recv - t_trigger) * 1000.0
                latencies.append(rt_latency_ms)

                self.assertTrue(response_bytes.startswith(b"RIFF"), "Must receive valid WAV file")
                self.assertGreater(len(response_bytes), 44)

        p50 = float(np.percentile(latencies, 50))
        p90 = float(np.percentile(latencies, 90))
        p99 = float(np.percentile(latencies, 99))
        print(f"\n[Test 2.1] WebSocket Sequential Streaming (15 turns): P50={p50:.1f}ms, P90={p90:.1f}ms, P99={p99:.1f}ms, Mean={np.mean(latencies):.1f}ms")
        self.assertLess(p50, 1500.0)
        self.assertLess(p90, 1500.0)
        self.assertLess(p99, 1500.0)

    def test_06_websocket_events_mode_streaming(self):
        """
        Benchmark WebSocket in events mode (?events=true&format=pcm) over 10 utterances.
        Verify both JSON metadata frame and binary PCM audio frame arrive with correct latency.
        """
        latencies: List[float] = []
        reported_latencies: List[float] = []
        num_turns = 10

        with self.client.websocket_connect("/ws/s2s_live?events=true&format=pcm") as ws:
            for _ in range(num_turns):
                for _ in range(3):
                    ws.send_bytes(make_pcm_chunk(is_speech=True, duration_s=0.3))
                ws.send_bytes(make_pcm_chunk(is_speech=False, duration_s=0.3))

                t_trigger = time.perf_counter()
                ws.send_bytes(make_pcm_chunk(is_speech=False, duration_s=0.3))

                # Receive JSON telemetry
                json_frame = ws.receive_json()
                # Receive raw PCM bytes
                pcm_bytes = ws.receive_bytes()
                t_recv = time.perf_counter()

                rt_latency = (t_recv - t_trigger) * 1000.0
                latencies.append(rt_latency)
                reported_latencies.append(json_frame["latency_ms"])

                self.assertEqual(json_frame["type"], "result")
                self.assertGreater(len(json_frame["vi_text"]), 0)
                self.assertGreater(len(json_frame["hmong_text"]), 0)
                self.assertGreater(len(pcm_bytes), 0)

        p50 = float(np.percentile(latencies, 50))
        p90 = float(np.percentile(latencies, 90))
        p99 = float(np.percentile(latencies, 99))
        print(f"[Test 2.2] WebSocket Events Mode (10 turns): P50={p50:.1f}ms, P90={p90:.1f}ms, P99={p99:.1f}ms, Mean={np.mean(latencies):.1f}ms")
        self.assertLess(p50, 1500.0)
        self.assertLess(p90, 1500.0)
        self.assertLess(p99, 1500.0)

    def test_07_websocket_concurrent_clients(self):
        """
        Benchmark WebSocket /ws/s2s_live under concurrent streaming sessions (4 concurrent clients,
        each performing 5 conversational turns = 20 total utterances under concurrent load).
        Verify P50, P90, P99 remain strictly < 1,500ms even under concurrency.
        """
        num_clients = 4
        turns_per_client = 5
        all_latencies: List[float] = []

        def client_worker(client_id: int) -> List[float]:
            worker_client = TestClient(app)
            worker_latencies: List[float] = []
            with worker_client.websocket_connect("/ws/s2s_live?format=wav") as ws:
                for turn in range(turns_per_client):
                    for _ in range(3):
                        ws.send_bytes(make_pcm_chunk(is_speech=True, duration_s=0.3, freq=300.0 + client_id * 50))
                    ws.send_bytes(make_pcm_chunk(is_speech=False, duration_s=0.3))

                    t_trigger = time.perf_counter()
                    ws.send_bytes(make_pcm_chunk(is_speech=False, duration_s=0.3))

                    resp = ws.receive_bytes()
                    t_recv = time.perf_counter()

                    self.assertTrue(resp.startswith(b"RIFF"))
                    worker_latencies.append((t_recv - t_trigger) * 1000.0)
            return worker_latencies

        with ThreadPoolExecutor(max_workers=num_clients) as pool:
            futures = [pool.submit(client_worker, cid) for cid in range(num_clients)]
            for fut in as_completed(futures):
                all_latencies.extend(fut.result())

        self.assertEqual(len(all_latencies), num_clients * turns_per_client)
        p50 = float(np.percentile(all_latencies, 50))
        p90 = float(np.percentile(all_latencies, 90))
        p99 = float(np.percentile(all_latencies, 99))
        print(f"[Test 2.3] WebSocket Concurrent Clients ({num_clients} clients x {turns_per_client} turns = {len(all_latencies)}): P50={p50:.1f}ms, P90={p90:.1f}ms, P99={p99:.1f}ms, Max={max(all_latencies):.1f}ms")
        self.assertLess(p50, 1500.0)
        self.assertLess(p90, 1500.0)
        self.assertLess(p99, 1500.0)

    # -----------------------------------------------------------------------
    # Challenge 3: Verify P50, P90, P99 Latency Strictly < 1,500ms
    # -----------------------------------------------------------------------
    def test_08_aggregate_latency_benchmark(self):
        """
        Execute comprehensive end-to-end streaming latency evaluation with 40 distinct utterances
        spanning short, medium, and long speech durations (0.6s to 2.4s).
        Compute P50, P90, P99 and assert strictly < 1,500ms.
        """
        speech_lengths = [2, 3, 4, 6, 8]  # Chunks corresponding to 0.6s, 0.9s, 1.2s, 1.8s, 2.4s
        latencies: List[float] = []

        with self.client.websocket_connect("/ws/s2s_live?format=pcm") as ws:
            for length in speech_lengths:
                for rep in range(8):  # 5 lengths * 8 reps = 40 samples
                    # Voiced chunks
                    for _ in range(length):
                        ws.send_bytes(make_pcm_chunk(is_speech=True, duration_s=0.3, freq=250.0 + (rep * 30)))
                    # Silence chunk 1
                    ws.send_bytes(make_pcm_chunk(is_speech=False, duration_s=0.3))
                    # Silence chunk 2 (trigger)
                    t0 = time.perf_counter()
                    ws.send_bytes(make_pcm_chunk(is_speech=False, duration_s=0.3))

                    pcm_bytes = ws.receive_bytes()
                    t1 = time.perf_counter()

                    lat_ms = (t1 - t0) * 1000.0
                    latencies.append(lat_ms)
                    self.assertGreater(len(pcm_bytes), 0)

        latencies_arr = np.array(latencies)
        mean_lat = float(np.mean(latencies_arr))
        p50 = float(np.percentile(latencies_arr, 50))
        p90 = float(np.percentile(latencies_arr, 90))
        p95 = float(np.percentile(latencies_arr, 95))
        p99 = float(np.percentile(latencies_arr, 99))
        min_lat = float(np.min(latencies_arr))
        max_lat = float(np.max(latencies_arr))

        print(f"\n[Test 3.1] Comprehensive Aggregate Streaming Latency (N={len(latencies)}):")
        print(f"  - Min:  {min_lat:.2f} ms")
        print(f"  - Mean: {mean_lat:.2f} ms")
        print(f"  - P50:  {p50:.2f} ms")
        print(f"  - P90:  {p90:.2f} ms")
        print(f"  - P95:  {p95:.2f} ms")
        print(f"  - P99:  {p99:.2f} ms")
        print(f"  - Max:  {max_lat:.2f} ms")

        # Formal acceptance assertions
        self.assertLess(p50, 1500.0, f"P50 latency {p50}ms must be strictly < 1500ms")
        self.assertLess(p90, 1500.0, f"P90 latency {p90}ms must be strictly < 1500ms")
        self.assertLess(p99, 1500.0, f"P99 latency {p99}ms must be strictly < 1500ms")

    # -----------------------------------------------------------------------
    # Challenge 4: Verify TTS RTF Strictly < 1.0
    # -----------------------------------------------------------------------
    def test_09_tts_rtf_comprehensive_benchmark(self):
        """
        Benchmark HmongVITS RTF (Real-Time Factor) across 30 diverse Hmong Daw RPA sentences
        covering short, medium, and long texts, all 8 lexical tone phonemes (-b, -m, -d, -j, -v, -s, -g, mid),
        and various synthesis speed factors (0.8x, 1.0x, 1.2x, 1.5x).
        Verify that RTF is strictly < 1.0 across all samples and percentiles.
        """
        test_sentences = [
            # Short phrases & greetings (1-3 words)
            "nyob zoo",
            "ua tsaug",
            "sib ntsib dua",
            "zoo siab",
            "mus zoo koj",
            "nyob zoo sawv ntxov",
            "koj tuaj los",
            "kuv yog hmoob",
            "hnub no zoo",
            "hais lus hmoob",
            # Medium sentences (4-8 words)
            "hnub no huab cua zoo heev peb mus ua si",
            "kuv xav kawm hais lus hmoob dawb zoo heev",
            "me nyuam yaus mus kawm ntawv tom tsev kawm",
            "peb sawv daws nyob ua ke muaj kev lom zem",
            "niam thiab txiv ua teb khwv heev nyob roob siab",
            "dej ntws ntshiab heev nyob hauv hav zoov",
            "hmoob dawb muaj ntau yam kab lis kev cai zoo nkauj",
            "lub tsev no loj heev thiab zoo nkauj kawg nkaus",
            "nag hmo nag los hnyav heev dej nyab kev",
            "peb yuav tsum sib pab thiab sib hlub",
            # Long sentences & complex RPA phonetics (9+ words)
            "nyob rau lub caij nplooj ntoos hlav txhua yam nroj tsuag pib hlav ntsuab xiab",
            "cov laus qhia rau cov hluas tias yuav tsum paub txog peb lub hauv paus",
            "hmoob cov lus muaj yim lub suab txawv uas yog b m d j v s g thiab suab nruab nrab",
            "thaum mus txog saum roob peb thiaj pom lub zos nyob nram hav zoo nkauj kawg nkaus",
            "lub tshuab no pab peb txhais lus ntawm hais lus việt mus rau lus hmoob dawb sai heev",
            "peb muaj kev zoo siab tau tuaj koom lub rooj sib tham txog kev tshawb fawb no",
            "txhua tus tib neeg yug los muaj kev ywj pheej thiab muaj txoj cai sib luag",
            "hmoob cov paj ntaub muaj ntau yam qauv zoo nkauj uas niam tais tau qhia tseg",
            "yog tias koj xav paub ntxiv koj tuaj yeem nug peb txhua lub sij hawm",
            "txoj kev kawm ntawv yog qhov tseem ceeb tshaj plaws rau peb lub neej yav tom ntej"
        ]

        speeds = [0.8, 1.0, 1.2, 1.5]
        rtf_records: List[Dict[str, Any]] = []
        rtf_values: List[float] = []

        # Warm-up pass
        self.tts.synthesize("nyob zoo")

        for idx, text in enumerate(test_sentences):
            speed = speeds[idx % len(speeds)]
            t0 = time.perf_counter()
            audio = self.tts.synthesize(text, speed=speed)
            t1 = time.perf_counter()

            synth_time_s = t1 - t0
            audio_dur_s = len(audio) / 16000.0
            self.assertGreater(audio_dur_s, 0.0, f"Synthesized audio for '{text}' must have non-zero duration")
            
            rtf = synth_time_s / audio_dur_s
            rtf_values.append(rtf)
            rtf_records.append({
                "text": text,
                "speed": speed,
                "audio_len_samples": len(audio),
                "audio_dur_s": round(audio_dur_s, 3),
                "synth_time_ms": round(synth_time_s * 1000.0, 2),
                "rtf": round(rtf, 4)
            })

        rtf_arr = np.array(rtf_values)
        mean_rtf = float(np.mean(rtf_arr))
        p50_rtf = float(np.percentile(rtf_arr, 50))
        p90_rtf = float(np.percentile(rtf_arr, 90))
        p95_rtf = float(np.percentile(rtf_arr, 95))
        p99_rtf = float(np.percentile(rtf_arr, 99))
        min_rtf = float(np.min(rtf_arr))
        max_rtf = float(np.max(rtf_arr))

        print(f"\n[Test 4.1] HmongVITS RTF Benchmark Results (N={len(rtf_values)}):")
        print(f"  - Min RTF:  {min_rtf:.4f}")
        print(f"  - Mean RTF: {mean_rtf:.4f}")
        print(f"  - P50 RTF:  {p50_rtf:.4f}")
        print(f"  - P90 RTF:  {p90_rtf:.4f}")
        print(f"  - P95 RTF:  {p95_rtf:.4f}")
        print(f"  - P99 RTF:  {p99_rtf:.4f}")
        print(f"  - Max RTF:  {max_rtf:.4f}")

        # Assert every single item satisfies RTF < 1.0
        self.assertLess(max_rtf, 1.0, f"Max RTF {max_rtf:.4f} exceeds 1.0 threshold!")
        self.assertLess(p99_rtf, 1.0, f"P99 RTF {p99_rtf:.4f} exceeds 1.0 threshold!")
        self.assertLess(p50_rtf, 1.0, f"P50 RTF {p50_rtf:.4f} exceeds 1.0 threshold!")


if __name__ == "__main__":
    unittest.main()
