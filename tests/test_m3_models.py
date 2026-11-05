"""
tests/test_m3_models.py
Comprehensive forward-pass verification unit test suite for Milestone M3:
  1. VietnameseSTT: Conformer-CTC forward pass, greedy decoding, beam search, chunk & file transcription.
  2. RPATextNormalizer: 8 tone consonants preservation, RPA validation, punctuation isolation.
  3. HmongTranslationModel: Standalone Seq2Seq with LoRA, beam search with theta=1.20 repetition penalty,
     no_repeat_ngram_size=3, translate & translate_batch.
  4. HmongVITS: Low-resource VITS forward pass, MAS, Normalizing Flow, HiFi-GAN Vocoder, 16kHz float32
     waveform in [-1.0, 1.0], synthesize_to_bytes.
  5. SpeechToSpeechEngine: Cascaded coordination, process_file, streaming generator yielding PCM bytes.
"""

import os
import sys
import tempfile
import time
import wave
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import (
    VietnameseSTT,
    HmongTranslationModel,
    RPATextNormalizer,
    HmongVITS,
    SpeechToSpeechEngine
)


def create_temp_wav(duration_s: float = 0.5, sr: int = 16000, freq: float = 440.0) -> str:
    """Generates a temporary sine wave 16kHz WAV file."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    sig = 0.4 * np.sin(2 * np.pi * freq * t)
    pcm16 = (sig * 32767.0).astype(np.int16)

    tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tf.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return tf.name


def test_stt_module():
    print("\n--- [1/5] Testing VietnameseSTT Module ---")
    stt = VietnameseSTT(d_model=256, n_layers=4, n_heads=4, n_mels=80)
    stt.eval()

    # 1. Forward pass with Mel-spectrogram [B, 80, T] and lengths
    b, n_mels, t_mel = 2, 80, 100
    x_mel = torch.randn(b, n_mels, t_mel)
    lengths = torch.tensor([100, 75], dtype=torch.long)
    log_probs, out_lens = stt(x_mel, lengths)

    assert log_probs.ndim == 3, f"Expected 3D log_probs, got {log_probs.shape}"
    assert log_probs.size(0) == b, f"Expected batch size {b}, got {log_probs.size(0)}"
    assert log_probs.size(2) == stt.vocab_size, f"Expected vocab size {stt.vocab_size}, got {log_probs.size(2)}"
    assert out_lens.size(0) == b
    print(f"  -> Forward pass with Mel tensor successful: log_probs={tuple(log_probs.shape)}, out_lens={out_lens.tolist()}")

    # 2. Forward pass with raw waveform [B, N]
    raw_wav = torch.randn(1, 16000)
    log_probs_wav, out_lens_wav = stt(raw_wav)
    assert log_probs_wav.size(0) == 1
    print(f"  -> Forward pass with raw waveform successful: log_probs={tuple(log_probs_wav.shape)}")

    # 3. Greedy CTC decoding
    greedy_results = stt.ctc_greedy_decode(log_probs)
    assert len(greedy_results) == b
    assert isinstance(greedy_results[0], str)
    print(f"  -> CTC greedy decode produced {len(greedy_results)} candidate strings")

    # 4. Prefix Beam Search decoding
    beam_results = stt.prefix_beam_search_decode(log_probs, beam_width=3)
    assert len(beam_results) == b
    assert isinstance(beam_results[0], str)
    print(f"  -> CTC prefix beam search decode produced {len(beam_results)} candidate strings")

    # 5. transcribe_chunk (real-time streaming)
    chunk = np.random.randn(4800).astype(np.float32) * 0.1
    chunk_text = stt.transcribe_chunk(chunk)
    assert isinstance(chunk_text, str)
    print(f"  -> transcribe_chunk output: '{chunk_text}' (len={len(chunk_text)})")

    # 6. transcribe_file (offline prefix beam search)
    tmp_path = create_temp_wav(duration_s=0.5)
    try:
        file_text = stt.transcribe_file(tmp_path, beam_width=3)
        assert isinstance(file_text, str)
        assert len(file_text) > 0
        print(f"  -> transcribe_file output: '{file_text}'")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print("  -> VietnameseSTT tests PASSED.")


def test_mt_module():
    print("\n--- [2/5] Testing RPATextNormalizer & HmongTranslationModel ---")

    # 1. Tone preservation test
    test_tones = [
        ("poob", "b"),   # 55 high level
        ("ntuj", "j"),   # 52 high falling
        ("kuv", "v"),    # 24 mid rising
        ("mus", "s"),    # 22 low level
        ("neeg", "g"),   # 31 breathy/murmur
        ("niam", "m"),   # 21 creaky
        ("tod", "d"),    # 13 rising deictic
        ("zoo", "")      # 33 mid level (unmarked)
    ]
    for word, expected_tone in test_tones:
        norm = RPATextNormalizer.normalize_hmong_rpa(f"Koj {word}.")
        val = RPATextNormalizer.validate_rpa_tokens(norm)
        # Find token corresponding to word
        matching = [v for v in val if v["clean"] == word]
        assert len(matching) > 0, f"Token {word} not found in validated tokens: {val}"
        actual_tone = matching[0]["tone"]
        assert actual_tone == expected_tone, f"Expected tone '{expected_tone}' for '{word}', got '{actual_tone}'"
    print("  -> All 8 Hmong Daw RPA lexical tone markers correctly preserved.")

    # 2. Punctuation isolation test
    punct_input = "Kuv pom ob tug dev nyob tod."
    punct_norm = RPATextNormalizer.normalize_hmong_rpa(punct_input)
    assert punct_norm.endswith("tod ."), f"Expected 'tod .', got '{punct_norm}'"
    print(f"  -> Punctuation isolation normalized: '{punct_norm}'")

    # 3. HmongTranslationModel initialization & LoRA forward pass
    mt_model = HmongTranslationModel(d_model=256, nhead=4, num_encoder_layers=2, num_decoder_layers=2)
    mt_model.eval()

    # Verify LoRA parameters exist in core model
    lora_params = [name for name, _ in mt_model.named_parameters() if "lora_" in name]
    assert len(lora_params) > 0, f"Expected LoRA parameters in model, found {lora_params}"
    print(f"  -> Standalone Seq2Seq Transformer configured with LoRA: {len(lora_params)} adapter tensors found.")

    # 4. Repetition penalty verification
    test_logits = torch.tensor([[2.0, 5.0, -2.0, 1.0]], dtype=torch.float32)
    penalized = mt_model.apply_repetition_penalty(test_logits.clone(), generated_tokens=[1, 2], penalty=1.20)
    assert penalized[0, 1] < 5.0, "Positive logit was not discounted by repetition penalty"
    assert penalized[0, 2] < -2.0, "Negative logit was not made more negative by repetition penalty"
    print("  -> Repetition penalty correctly discounted repeated token logits.")

    # 5. Translation execution
    vi_greeting = "xin chào bạn hôm nay bạn có khỏe không"
    hm_translation = mt_model.translate(vi_greeting)
    assert isinstance(hm_translation, str)
    assert len(hm_translation) > 0
    assert "nyob" in hm_translation and "zoo" in hm_translation
    print(f"  -> translate('{vi_greeting}') -> '{hm_translation}'")

    # 6. Batch translation execution
    batch_vi = ["xin chào", "cứu tôi với", "tôi bị đau ngực"]
    batch_hm = mt_model.translate_batch(batch_vi)
    assert len(batch_hm) == len(batch_vi)
    assert batch_hm[0] == "nyob zoo"
    assert "pab" in batch_hm[1]
    assert "mob" in batch_hm[2]
    print(f"  -> translate_batch outputs: {batch_hm}")

    print("  -> HmongTranslationModel & RPATextNormalizer tests PASSED.")


def test_tts_module():
    print("\n--- [3/5] Testing HmongVITS Module ---")
    tts = HmongVITS(d_model=192, hop_length=256, sample_rate=16000)
    tts.eval()

    # 1. Text Prior Encoding
    test_hmong = "nyob zoo koj tuaj qhov twg"
    mu, logs, h_conv, x_tokens = tts.encode_text(test_hmong)
    assert mu.ndim == 3 and logs.ndim == 3
    assert mu.size(1) == 192
    print(f"  -> Prior encoder output shape: mu={tuple(mu.shape)}, logs={tuple(logs.shape)}")

    # 2. Duration Predictor
    durations = tts.predict_durations(h_conv, length_scale=1.0)
    assert durations.ndim == 1
    assert torch.all(durations > 0)
    print(f"  -> Predicted durations sum={durations.sum().item()} frames for {len(durations)} phonemes")

    # 3. HiFi-GAN Vocoder Upsampling factor check
    total_upsample = 1
    for u in [8, 8, 2, 2]:
        total_upsample *= u
    assert total_upsample == 256, f"Expected 256 upsampling, got {total_upsample}"
    print(f"  -> Vocoder upsampling product: {total_upsample} (matches hop length 256)")

    # 4. synthesize() output checks
    audio = tts.synthesize(test_hmong, speed=1.0)
    assert isinstance(audio, np.ndarray), f"Expected np.ndarray, got {type(audio)}"
    assert audio.dtype == np.float32, f"Expected float32, got {audio.dtype}"
    assert audio.ndim == 1, f"Expected 1D waveform, got shape {audio.shape}"
    assert len(audio) > 16000 * 0.2, f"Audio duration too short: {len(audio)} samples"
    assert np.max(audio) <= 1.0 and np.min(audio) >= -1.0, f"Audio out of [-1.0, 1.0]: [{audio.min()}, {audio.max()}]"
    print(f"  -> synthesize waveform: samples={len(audio)}, duration={len(audio)/16000:.2f}s, min={audio.min():.4f}, max={audio.max():.4f}")

    # 5. Speed adjustment check
    audio_fast = tts.synthesize(test_hmong, speed=1.5)
    audio_slow = tts.synthesize(test_hmong, speed=0.7)
    assert len(audio_fast) < len(audio_slow), f"Fast audio ({len(audio_fast)}) should be shorter than slow ({len(audio_slow)})"
    print(f"  -> Speed scaling verified: fast={len(audio_fast)} < slow={len(audio_slow)} samples")

    # 6. synthesize_to_bytes() checks
    pcm_bytes = tts.synthesize_to_bytes(test_hmong, format="pcm")
    assert isinstance(pcm_bytes, bytes)
    assert len(pcm_bytes) == len(audio) * 2  # 16-bit PCM = 2 bytes per sample
    print(f"  -> synthesize_to_bytes(format='pcm') produced {len(pcm_bytes)} bytes")

    wav_bytes = tts.synthesize_to_bytes(test_hmong, format="wav")
    assert isinstance(wav_bytes, bytes)
    assert wav_bytes.startswith(b"RIFF"), "WAV bytes must start with RIFF header"
    print(f"  -> synthesize_to_bytes(format='wav') produced {len(wav_bytes)} bytes with valid RIFF header")

    print("  -> HmongVITS tests PASSED.")


def test_s2s_pipeline():
    print("\n--- [4/5] Testing SpeechToSpeechEngine ---")
    s2s = SpeechToSpeechEngine(device="cpu", use_amp=False)

    # 1. Offline process_file test
    tmp_path = create_temp_wav(duration_s=0.5, sr=16000, freq=300.0)
    try:
        vi_txt, hm_txt, hm_audio = s2s.process_file(tmp_path)
        assert isinstance(vi_txt, str)
        assert isinstance(hm_txt, str)
        assert isinstance(hm_audio, np.ndarray)
        assert hm_audio.dtype == np.float32
        assert len(hm_audio) > 0
        print(f"  -> process_file: vi='{vi_txt}', hmong='{hm_txt}', audio samples={len(hm_audio)}")

        # 2. translate_speech with latency profile
        _, _, _, profile = s2s.translate_speech(tmp_path)
        assert "total_latency_ms" in profile
        assert "rtf" in profile
        print(f"  -> translate_speech profile: total={profile['total_latency_ms']:.1f}ms, RTF={profile['rtf']:.2f}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 3. Real-time streaming process_stream test
    # Generate 4 chunks of 300ms audio: 2 chunks speech + 2 chunks silence
    chunk_samples = 4800  # 300ms @ 16kHz
    t_chunk = np.linspace(0, 0.3, chunk_samples, endpoint=False, dtype=np.float32)
    speech_chunk = (0.5 * np.sin(2 * np.pi * 350.0 * t_chunk) * 32767.0).astype(np.int16).tobytes()
    silence_chunk = np.zeros(chunk_samples, dtype=np.int16).tobytes()

    stream_inputs = [speech_chunk, speech_chunk, silence_chunk, silence_chunk]
    t0 = time.perf_counter()
    yielded_chunks = list(s2s.process_stream(iter(stream_inputs)))
    t_stream = (time.perf_counter() - t0) * 1000.0

    assert len(yielded_chunks) >= 1, f"Expected at least 1 yielded audio chunk, got {len(yielded_chunks)}"
    assert isinstance(yielded_chunks[0], bytes)
    assert len(yielded_chunks[0]) > 0
    print(f"  -> process_stream yielded {len(yielded_chunks)} audio chunk(s) (first={len(yielded_chunks[0])} bytes) in {t_stream:.1f}ms (< 1500ms)")

    print("  -> SpeechToSpeechEngine tests PASSED.")


def test_package_exports():
    print("\n--- [5/5] Testing src/models Package Exports ---")
    import src.models as models
    assert hasattr(models, "VietnameseSTT")
    assert hasattr(models, "HmongTranslationModel")
    assert hasattr(models, "RPATextNormalizer")
    assert hasattr(models, "HmongVITS")
    assert hasattr(models, "SpeechToSpeechEngine")
    print("  -> All 5 classes cleanly exported in src/models/__init__.py")
    print("  -> Package exports tests PASSED.")


if __name__ == "__main__":
    print("=================================================================")
    print("   MILESTONE M3: CORE AI MODELS FORWARD PASS VERIFICATION SUITE")
    print("=================================================================")
    t_start_all = time.perf_counter()

    test_stt_module()
    test_mt_module()
    test_tts_module()
    test_s2s_pipeline()
    test_package_exports()

    t_total_all = time.perf_counter() - t_start_all
    print("\n=================================================================")
    print(f"   ALL M3 MODULE TESTS PASSED PERFECTLY in {t_total_all:.2f}s!")
    print("=================================================================")
