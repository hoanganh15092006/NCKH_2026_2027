"""
scripts/__init__.py
Package initialization for training and evaluation pipeline scripts.
Provides centralized access to:
  - train_stt: Conformer-CTC speech recognition training pipeline
  - train_mt: Seq2Seq machine translation training pipeline with back-translation
  - train_tts: Hmong VITS multi-objective speech synthesis training pipeline
"""

__all__ = ["train_stt", "train_mt", "train_tts"]
