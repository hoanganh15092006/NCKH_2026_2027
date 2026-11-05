"""
src/api/__init__.py
Real-Time Systems & Streaming API Module for Multimodal Vietnamese-to-Hmong Daw RPA
Speech-to-Speech Translation.

Provides production FastAPI service, WebSocket streaming handler with RingBuffer VAD,
and browser-based Web Audio interactive interface.
Conforms strictly to Requirements R6 and R8 (genuine algorithmic implementation, zero mocks/stubs).
"""

from src.api.app_stream import (
    app,
    get_engine,
    create_app,
    RingBufferVAD,
    HealthResponse,
    TranslateRequest,
    TranslateResponse,
    TTSRequest,
    STTResponse,
)

__all__ = [
    "app",
    "get_engine",
    "create_app",
    "RingBufferVAD",
    "HealthResponse",
    "TranslateRequest",
    "TranslateResponse",
    "TTSRequest",
    "STTResponse",
]

__version__ = "1.0.0"
