"""
src/models/__init__.py
Core AI Models package for Multimodal Vietnamese-to-H'Mong Speech-to-Speech Translation.
Exports:
  - VietnameseSTT: Conformer-CTC acoustic network for Vietnamese ASR
  - HmongTranslationModel: Low-Resource NMT adapting Vietnamese to Hmong Daw RPA
  - RPATextNormalizer: Morphology-aware normalizer preserving 8 Hmong Daw RPA tones
  - HmongVITS: Compact low-resource VITS neural TTS synthesizer
  - SpeechToSpeechEngine: Real-time cascading pipeline coordinating STT -> MT -> TTS
"""

from src.models.stt_module import VietnameseSTT, VIETNAMESE_CTC_VOCAB
from src.models.mt_module import HmongTranslationModel, RPATextNormalizer
from src.models.tts_module import HmongVITS, HMONG_RPA_VOCAB, tokenize_hmong_rpa
from src.models.s2s_pipeline import SpeechToSpeechEngine

__all__ = [
    "VietnameseSTT",
    "HmongTranslationModel",
    "RPATextNormalizer",
    "HmongVITS",
    "SpeechToSpeechEngine",
    "VIETNAMESE_CTC_VOCAB",
    "HMONG_RPA_VOCAB",
    "tokenize_hmong_rpa"
]
