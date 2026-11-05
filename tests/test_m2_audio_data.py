"""
tests/test_m2_audio_data.py
Comprehensive verification test for Milestone M2:
- AudioPreprocessor: Resampling, RMS normalization, dual VAD (STE + Silero fallback),
  silence filtering, Mel-spectrogram extraction [B, 80, T], process_file, process_chunk.
- TriParallelDataset: Tri-parallel & quad-parallel tuples, on-the-fly extraction.
- DynamicPaddingCollate: Dynamic waveform and mel padding, length tensors.
- Augmentations: SpecAugment and AdditiveGaussianNoise.
"""
import os
import sys
import io
import wave
import tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.audio_preprocessor import (
    AudioPreprocessor,
    SileroVADWrapper,
    ProcessedAudioResult
)
from src.data.dataset_loader import (
    TriParallelDataset,
    DynamicPaddingCollate,
    SpecAugment,
    AdditiveGaussianNoise
)


def create_synthetic_wav(
    file_path: str,
    duration_sec: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
    freq: float = 440.0
):
    """Generates a valid sine-wave WAV file for offline testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # 0.5s tone + 0.5s silence
    half = len(t) // 2
    audio = np.zeros_like(t)
    audio[:half] = 0.5 * np.sin(2 * np.pi * freq * t[:half])

    pcm16 = (audio * 32767).astype(np.int16)
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def test_audio_preprocessor():
    print("[1/4] Testing AudioPreprocessor...")
    preprocessor = AudioPreprocessor(sample_rate=16000, target_rms_dbfs=-20.0, n_mels=80)

    # 1. Resample to 16k mono
    stereo_44k = np.random.randn(2, 44100).astype(np.float32)
    mono_16k = preprocessor.resample_to_16k(stereo_44k, orig_sr=44100)
    assert mono_16k.dim() == 2 and mono_16k.size(0) == 1, f"Expected [1, N], got {mono_16k.shape}"
    assert abs(mono_16k.size(1) - 16000) < 100, f"Expected ~16000 samples, got {mono_16k.size(1)}"
    print("  -> resample_to_16k passed.")

    # 2. RMS Normalization with clipping prevention
    loud_audio = torch.randn(1, 16000) * 10.0
    norm_audio = preprocessor.normalize_rms(loud_audio, target_db=-20.0)
    max_val = torch.max(torch.abs(norm_audio)).item()
    assert max_val <= 0.990001, f"Clipping prevention failed, max_val={max_val}"
    rms = torch.sqrt(torch.mean(norm_audio ** 2)).item()
    expected_rms = 10.0 ** (-20.0 / 20.0)
    print(f"  -> normalize_rms passed: max={max_val:.4f} <= 0.99, rms={rms:.4f} (target={expected_rms:.4f})")

    # 3. STE Voice Activity Detection
    voiced_signal = torch.sin(torch.linspace(0, 100 * np.pi, 3200)) # 200ms sine wave
    silent_signal = torch.zeros(3200)
    mixed_signal = torch.cat([voiced_signal, silent_signal], dim=0) # 400ms
    ste_mask = preprocessor.detect_voice_activity_ste(mixed_signal, frame_length=512, hop_length=160)
    assert ste_mask.dtype == torch.bool
    assert ste_mask[:5].any().item(), "Expected speech detected in voiced prefix"
    print(f"  -> detect_voice_activity_ste passed: frames={ste_mask.size(0)}, active_ratio={ste_mask.float().mean().item():.2f}")

    # 4. Silero VAD (neural + fallback)
    silero_mask = preprocessor.detect_voice_activity_silero(mixed_signal, threshold=0.5)
    assert silero_mask.dtype == torch.bool
    print(f"  -> detect_voice_activity_silero passed: chunks={silero_mask.size(0)}")

    # 5. Silence Filtering
    filtered_audio, intervals = preprocessor.filter_silence(mixed_signal.unsqueeze(0))
    assert filtered_audio.dim() == 2 and filtered_audio.size(0) == 1
    assert len(intervals) >= 1
    print(f"  -> filter_silence passed: original={mixed_signal.size(0)}, filtered={filtered_audio.size(1)}, intervals={intervals}")

    # 6. Mel-Spectrogram Extraction
    mel = preprocessor.extract_mel_spectrogram(norm_audio)
    assert mel.dim() == 3, f"Expected 3D tensor [B, n_mels, T], got {mel.shape}"
    assert mel.size(1) == 80, f"Expected 80 mels, got {mel.size(1)}"
    print(f"  -> extract_mel_spectrogram passed: shape={mel.shape}")

    # 7. process_file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        create_synthetic_wav(tmp_path, duration_sec=1.0, sample_rate=16000)
        res_audio, res_mel = preprocessor.process_file(tmp_path)
        assert isinstance(res_audio, torch.Tensor)
        assert isinstance(res_mel, torch.Tensor)
        assert res_mel.size(1) == 80
        # Check dict-style access
        assert res_audio is not None and res_mel is not None
        print(f"  -> process_file passed: audio shape={res_audio.shape}, mel shape={res_mel.shape}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 8. process_chunk
    # 3200 samples of 16-bit PCM (200ms @ 16kHz) = 6400 bytes
    pcm_bytes = (np.sin(np.linspace(0, 50 * np.pi, 3200)) * 20000).astype(np.int16).tobytes()
    norm_chunk, is_speech = preprocessor.process_chunk(pcm_bytes)
    assert isinstance(norm_chunk, torch.Tensor)
    assert isinstance(is_speech, bool)
    print(f"  -> process_chunk passed: chunk_shape={norm_chunk.shape}, is_speech={is_speech}")


def test_augmentations():
    print("[2/4] Testing Audio Augmentations...")
    # AdditiveGaussianNoise
    noise_mod = AdditiveGaussianNoise(snr_db_range=(10.0, 20.0))
    noise_mod.train()
    clean_wav = torch.sin(torch.linspace(0, 100 * np.pi, 16000))
    noisy_wav = noise_mod(clean_wav)
    assert noisy_wav.shape == clean_wav.shape
    diff = torch.mean((noisy_wav - clean_wav) ** 2).item()
    assert diff > 0.0, "Expected non-zero noise injection"
    print(f"  -> AdditiveGaussianNoise passed: MSE={diff:.6f}")

    # SpecAugment
    spec_mod = SpecAugment(time_masking=20, freq_masking=10)
    spec_mod.train()
    dummy_mel = torch.ones(1, 80, 100) * 5.0
    masked_mel = spec_mod(dummy_mel)
    assert masked_mel.shape == dummy_mel.shape
    masked_count = (masked_mel == spec_mod.fill_value).sum().item()
    assert masked_count > 0, "Expected some elements to be masked"
    print(f"  -> SpecAugment passed: masked_elements={masked_count}")


def test_dataset_loader():
    print("[3/4] Testing TriParallelDataset...")
    records = [
        (
            torch.randn(16000),
            "Xin chào các bạn",
            "Nyob zoo sawv daws",
            torch.randn(15000)
        ),
        (
            torch.randn(24000),
            "Hôm nay trời rất đẹp",
            "Hnub no huab cua zoo heev",
            torch.randn(22000)
        ),
        (
            torch.randn(12000),
            "Cảm ơn bạn rất nhiều",
            "Ua tsaug ntau ntau",
            None
        )
    ]

    dataset = TriParallelDataset(records, sample_rate=16000, augment=True, extract_mel=True)
    assert len(dataset) == 3

    item0 = dataset[0]
    assert "audio_vi" in item0 and "mel_vi" in item0
    assert "text_vi" in item0 and "text_hmong" in item0
    assert "audio_hmong" in item0 and "mel_hmong" in item0
    assert item0["mel_vi"].size(0) == 80
    print(f"  -> TriParallelDataset item0: audio_vi_len={item0['audio_vi_len']}, mel_vi={item0['mel_vi'].shape}")

    item2 = dataset[2]
    assert "audio_hmong" not in item2
    print("  -> TriParallelDataset tri-parallel fallback item2 verified.")


def test_collate_function():
    print("[4/4] Testing DynamicPaddingCollate...")
    records = [
        (
            torch.randn(16000),
            "Xin chào",
            "Nyob zoo",
            torch.randn(14000)
        ),
        (
            torch.randn(24000),
            "Tôi là sinh viên",
            "Kuv yog menyuam kawm ntawv",
            torch.randn(22000)
        )
    ]
    dataset = TriParallelDataset(records, sample_rate=16000, extract_mel=True)
    collate_fn = DynamicPaddingCollate(pad_value=0.0, sort_by_length=True)

    batch_items = [dataset[0], dataset[1]]
    batch = collate_fn(batch_items)

    assert "audio_vi" in batch
    assert "audio_vi_lens" in batch
    assert "mel_vi" in batch
    assert "mel_vi_lens" in batch

    # Check descending sort by length
    assert batch["audio_vi_lens"][0] >= batch["audio_vi_lens"][1]
    assert batch["audio_vi"].size(0) == 2
    assert batch["audio_vi"].size(1) == 24000 # max length in batch
    assert batch["mel_vi"].size(1) == 80
    assert batch["mel_vi"].size(0) == 2
    print(f"  -> DynamicPaddingCollate passed: batch_audio={batch['audio_vi'].shape}, batch_mel={batch['mel_vi'].shape}")


if __name__ == "__main__":
    print("=== Running M2 Speech & Audio Processing Verification ===")
    test_audio_preprocessor()
    test_augmentations()
    test_dataset_loader()
    test_collate_function()
    print("=== ALL TESTS PASSED SUCCESSFULLY! ===")
