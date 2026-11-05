"""
tests/test_api_stream.py
Comprehensive Test Suite for Real-Time Streaming API & WebSocket Service (src/api/app_stream.py).
Tests:
  - System health endpoint: GET /api/v1/health
  - Vietnamese STT endpoint: POST /api/v1/stt (multipart and raw bytes)
  - Hmong Daw RPA translation endpoint: POST /api/v1/translate
  - Hmong VITS TTS synthesis endpoint: POST /api/v1/tts
  - Cascaded S2S translation endpoint: POST /api/v1/s2s with X-Vi-Text and X-Hmong-Text headers
  - WebSocket live streaming: /ws/s2s_live (binary PCM chunks and events mode)
  - Responsive HTML5 Web Audio client: GET /
  - RingBufferVAD unit functionality
"""

import io
import time
import wave
import unittest
import numpy as np
import torch
from fastapi.testclient import TestClient

from src.api.app_stream import (
    app,
    get_engine,
    RingBufferVAD,
    make_safe_header,
    parse_audio_bytes,
)


def create_synthetic_wav_bytes(
    duration_s: float = 0.6,
    sample_rate: int = 16000,
    frequency: float = 300.0,
    amplitude: float = 0.5
) -> bytes:
    """Generates synthetic 16kHz mono 16-bit PCM WAV bytes."""
    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, num_samples, endpoint=False, dtype=np.float32)
    waveform = amplitude * np.sin(2 * np.pi * frequency * t)
    pcm16 = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def create_pcm_chunk(
    is_speech: bool = True,
    sample_rate: int = 16000,
    duration_s: float = 0.3
) -> bytes:
    """Creates a 300ms 16-bit linear PCM chunk (active speech or silence)."""
    num_samples = int(duration_s * sample_rate)
    if is_speech:
        t = np.linspace(0, duration_s, num_samples, endpoint=False, dtype=np.float32)
        waveform = 0.5 * np.sin(2 * np.pi * 350.0 * t)
    else:
        waveform = np.zeros(num_samples, dtype=np.float32)

    pcm16 = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm16.tobytes()


class TestStreamingAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Force CPU device for deterministic, ultra-fast test execution
        cls.engine = get_engine(device="cpu", use_amp=False)
        app.state.engine = cls.engine
        cls.client = TestClient(app)

    def test_01_health_endpoint(self):
        """Verify GET /api/v1/health returns complete status and model readiness."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["models_loaded"])
        self.assertEqual(data["sample_rate"], 16000)
        self.assertIn("components", data)
        self.assertIn("stt_module", data["components"])
        self.assertIn("mt_module", data["components"])
        self.assertIn("tts_module", data["components"])
        self.assertIn("s2s_pipeline", data["components"])
        # Verify request timing middleware
        self.assertIn("x-process-time", response.headers)

    def test_02_translate_endpoint_valid(self):
        """Verify POST /api/v1/translate returns translated Hmong Daw RPA text."""
        payload = {"text": "xin chào bạn hôm nay bạn có khỏe không"}
        response = self.client.post("/api/v1/translate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("translated_text", data)
        self.assertIsInstance(data["translated_text"], str)
        self.assertGreater(len(data["translated_text"]), 0)

    def test_03_translate_endpoint_validation_error(self):
        """Verify POST /api/v1/translate rejects invalid or empty JSON payloads."""
        response = self.client.post("/api/v1/translate", json={"text": ""})
        self.assertEqual(response.status_code, 422)

    def test_04_tts_endpoint_valid(self):
        """Verify POST /api/v1/tts returns valid 16kHz audio/wav bytes stream."""
        payload = {"text": "nyob zoo koj puas nyob zoo", "speed": 1.0}
        response = self.client.post("/api/v1/tts", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertIn("Content-Disposition", response.headers)
        # Verify WAV RIFF header
        self.assertTrue(response.content.startswith(b"RIFF"))
        self.assertGreater(len(response.content), 44)

    def test_05_stt_endpoint_multipart_file(self):
        """Verify POST /api/v1/stt transcribes uploaded multipart WAV file."""
        wav_bytes = create_synthetic_wav_bytes(duration_s=0.5)
        files = {"file": ("test_sample.wav", wav_bytes, "audio/wav")}
        response = self.client.post("/api/v1/stt", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("text", data)
        self.assertIsInstance(data["text"], str)
        self.assertGreater(len(data["text"]), 0)

    def test_06_stt_endpoint_raw_bytes_body(self):
        """Verify POST /api/v1/stt transcribes raw audio bytes from request body."""
        wav_bytes = create_synthetic_wav_bytes(duration_s=0.5)
        headers = {"Content-Type": "audio/wav"}
        response = self.client.post("/api/v1/stt", content=wav_bytes, headers=headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("text", data)
        self.assertGreater(len(data["text"]), 0)

    def test_07_stt_endpoint_empty_payload_error(self):
        """Verify POST /api/v1/stt returns 400 when empty body is sent."""
        response = self.client.post("/api/v1/stt", content=b"")
        self.assertEqual(response.status_code, 400)

    def test_08_s2s_cascaded_endpoint(self):
        """
        Verify POST /api/v1/s2s executes end-to-end Cascaded S2S:
        Audio (Vi) -> STT -> MT -> TTS -> Audio (Hmong Daw)
        Checks audio/wav payload, and X-Vi-Text and X-Hmong-Text headers.
        """
        wav_bytes = create_synthetic_wav_bytes(duration_s=0.6)
        files = {"file": ("vietnamese_input.wav", wav_bytes, "audio/wav")}
        response = self.client.post("/api/v1/s2s", files=files)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertTrue(response.content.startswith(b"RIFF"))
        self.assertGreater(len(response.content), 44)

        # Check response headers
        self.assertIn("x-vi-text", response.headers)
        self.assertIn("x-hmong-text", response.headers)
        self.assertGreater(len(response.headers["x-vi-text"]), 0)
        self.assertGreater(len(response.headers["x-hmong-text"]), 0)

    def test_09_index_html_endpoint(self):
        """Verify GET / returns complete HTML5 + CSS3 + Web Audio API interface."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        content = response.text
        self.assertIn("<!DOCTYPE html>", content)
        self.assertIn("waveformCanvas", content)
        self.assertIn("ws/s2s_live", content)
        self.assertIn("recordBtn", content)
        self.assertIn("stopBtn", content)
        self.assertIn("viOutput", content)
        self.assertIn("hmOutput", content)

    def test_10_websocket_s2s_live_binary_mode(self):
        """
        Verify WebSocket /ws/s2s_live consumes binary 16kHz PCM chunks,
        triggers RingBuffer VAD endpointing, and streams synthesized audio bytes back.
        """
        with self.client.websocket_connect("/ws/s2s_live?format=wav") as ws:
            # 1. Send 3 voiced chunks (900ms speech)
            for _ in range(3):
                speech_chunk = create_pcm_chunk(is_speech=True)
                ws.send_bytes(speech_chunk)

            # 2. Send 2 silence chunks (600ms silence >= 400ms min_silence) to trigger endpoint
            for _ in range(2):
                silence_chunk = create_pcm_chunk(is_speech=False)
                ws.send_bytes(silence_chunk)

            # 3. Receive synthesized Hmong audio bytes
            audio_bytes = ws.receive_bytes()
            self.assertIsInstance(audio_bytes, bytes)
            self.assertGreater(len(audio_bytes), 0)
            self.assertTrue(audio_bytes.startswith(b"RIFF"))

    def test_11_websocket_s2s_live_events_mode(self):
        """
        Verify WebSocket /ws/s2s_live?events=true yields structured JSON event dictionaries
        with transcription, translation, and latency metrics alongside audio bytes.
        """
        with self.client.websocket_connect("/ws/s2s_live?events=true&format=wav") as ws:
            for _ in range(3):
                ws.send_bytes(create_pcm_chunk(is_speech=True))
            for _ in range(2):
                ws.send_bytes(create_pcm_chunk(is_speech=False))

            # Receive JSON event frame
            event_data = ws.receive_json()
            self.assertEqual(event_data.get("type"), "result")
            self.assertIn("vi_text", event_data)
            self.assertIn("hmong_text", event_data)
            self.assertIn("latency_ms", event_data)
            self.assertIn("audio_base64", event_data)
            self.assertGreater(len(event_data["vi_text"]), 0)
            self.assertGreater(len(event_data["hmong_text"]), 0)

            # Receive binary audio frame
            audio_bytes = ws.receive_bytes()
            self.assertTrue(audio_bytes.startswith(b"RIFF"))

    def test_12_ring_buffer_vad_unit(self):
        """Unit test on RingBufferVAD logic: speech onset, buffering, and silence endpointing."""
        vad = RingBufferVAD(
            sample_rate=16000,
            chunk_size_ms=300,
            min_silence_ms=400,
            min_speech_ms=200
        )
        preprocessor = self.engine.preprocessor

        # 1. Silence chunk should not trigger speech
        silence = create_pcm_chunk(is_speech=False)
        out = vad.add_chunk(silence, preprocessor)
        self.assertIsNone(out)
        self.assertFalse(vad.in_speech)

        # 2. Voiced chunk should transition to in_speech
        speech = create_pcm_chunk(is_speech=True)
        out = vad.add_chunk(speech, preprocessor)
        self.assertIsNone(out)
        self.assertTrue(vad.in_speech)

        # 3. Another voiced chunk
        out = vad.add_chunk(speech, preprocessor)
        self.assertIsNone(out)

        # 4. Silence chunk 1 (300ms < 400ms)
        out = vad.add_chunk(silence, preprocessor)
        self.assertIsNone(out)
        self.assertTrue(vad.in_speech)

        # 5. Silence chunk 2 (600ms >= 400ms) -> triggers endpoint!
        utterance = vad.add_chunk(silence, preprocessor)
        self.assertIsNotNone(utterance)
        self.assertIsInstance(utterance, bytes)
        self.assertGreater(len(utterance), 0)
        self.assertFalse(vad.in_speech)

    def test_text_to_speech_translation_endpoint(self):
        """Verify dedicated POST /api/v1/t2s translates text and returns WAV audio with headers."""
        payload = {"text": "bác sĩ cho tôi hỏi bệnh viện ở đâu", "speed": 1.0}
        response = self.client.post("/api/v1/t2s", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertTrue(response.content.startswith(b"RIFF"))
        self.assertGreater(len(response.content), 44)
        self.assertIn("x-vi-text", response.headers)
        self.assertIn("x-hmong-text", response.headers)
        self.assertIn("Content-Disposition", response.headers)

        # Validation error on empty text
        err_res = self.client.post("/api/v1/t2s", json={"text": ""})
        self.assertEqual(err_res.status_code, 422)

        # Blank whitespace text returns 400
        blank_res = self.client.post("/api/v1/t2s", json={"text": "   "})
        self.assertEqual(blank_res.status_code, 400)

    def test_s2s_direct_text_input(self):
        """Verify POST /api/v1/s2s directly accepts text input (JSON and Form), skipping STT."""
        # 1. Direct JSON payload
        json_payload = {"text": "xin chào bạn hôm nay bạn có khỏe không"}
        res_json = self.client.post("/api/v1/s2s", json=json_payload)
        self.assertEqual(res_json.status_code, 200)
        self.assertEqual(res_json.headers["content-type"], "audio/wav")
        self.assertTrue(res_json.content.startswith(b"RIFF"))
        self.assertGreater(len(res_json.content), 44)
        self.assertIn("x-vi-text", res_json.headers)
        self.assertIn("x-hmong-text", res_json.headers)

        # 2. Form data payload
        form_payload = {"text": "tôi cần mua thuốc cảm cúm và hạ sốt"}
        res_form = self.client.post("/api/v1/s2s", data=form_payload)
        self.assertEqual(res_form.status_code, 200)
        self.assertEqual(res_form.headers["content-type"], "audio/wav")
        self.assertTrue(res_form.content.startswith(b"RIFF"))
        self.assertGreater(len(res_form.content), 44)
        self.assertIn("x-vi-text", res_form.headers)
        self.assertIn("x-hmong-text", res_form.headers)

        # 3. Empty text without audio returns 400
        res_empty = self.client.post("/api/v1/s2s", json={"text": ""})
        self.assertEqual(res_empty.status_code, 400)

    def test_index_html_has_text_input_elements(self):
        """Verify INDEX_HTML contains responsive text input card, buttons, and native audio player."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html_content = response.text

        # Text input elements
        self.assertIn('id="viTextInput"', html_content)
        self.assertIn('id="translateTextBtn"', html_content)
        self.assertIn('id="clearTextBtn"', html_content)

        # Web Audio & native accessible player
        self.assertIn('id="nativeAudioPlayer"', html_content)
        self.assertIn('id="autoplayNotice"', html_content)
        self.assertIn('id="replayBtn"', html_content)

        # Autoplay mitigation and unlock logic
        self.assertIn("unlockAudioPlayback", html_content)
        self.assertIn("triggerTextTranslation", html_content)
        self.assertIn("btn-replay-pulse", html_content)


if __name__ == "__main__":
    unittest.main()
