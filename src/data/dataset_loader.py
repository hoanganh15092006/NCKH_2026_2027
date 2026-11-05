"""
src/data/dataset_loader.py
Tri-parallel and Quad-parallel PyTorch Dataset and Dynamic Padding DataLoader
supporting audio_vi, text_vi, text_hmong, and audio_hmong corpora,
with on-the-fly feature extraction, SpecAugment, and additive SNR Gaussian noise.
"""
from typing import List, Dict, Tuple, Optional, Any, Union
import json
import csv
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio

from src.data.audio_preprocessor import AudioPreprocessor


class AdditiveGaussianNoise(nn.Module):
    """
    Audio augmentation injecting additive white Gaussian noise calibrated
    to a specified Signal-to-Noise Ratio (SNR) range in decibels (dB).
    """
    def __init__(self, snr_db_range: Tuple[float, float] = (10.0, 20.0)):
        super().__init__()
        self.snr_db_range = snr_db_range

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Input: [N], [1, N], or [B, N] waveform tensor.
        Output: Noisy waveform tensor of same shape.
        """
        if not self.training:
            return wav

        snr_db = random.uniform(self.snr_db_range[0], self.snr_db_range[1])
        signal_power = torch.mean(wav ** 2)

        if signal_power <= 1e-9:
            return wav

        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = torch.randn_like(wav) * torch.sqrt(noise_power)
        noisy_wav = wav + noise

        # Clipping prevention
        max_val = torch.max(torch.abs(noisy_wav))
        if max_val > 0.99:
            noisy_wav = noisy_wav * (0.99 / max_val)

        return noisy_wav


class SpecAugment(nn.Module):
    """
    SpecAugment module applying Time Masking and Frequency Masking
    to Log Mel-spectrogram tensors [B, n_mels, T] or [n_mels, T].
    """
    def __init__(
        self,
        time_masking: int = 35,
        freq_masking: int = 15,
        time_mask_param: Optional[int] = None,
        freq_mask_param: Optional[int] = None,
        fill_value: float = -11.5129 # log(1e-5) floor value
    ):
        super().__init__()
        self.time_mask_param = time_mask_param if time_mask_param is not None else time_masking
        self.freq_mask_param = freq_mask_param if freq_mask_param is not None else freq_masking
        self.fill_value = fill_value

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        mel_spectrogram: Tensor of shape [n_mels, T] or [B, n_mels, T].
        """
        if not self.training:
            return mel_spectrogram

        was_2d = mel_spectrogram.dim() == 2
        if was_2d:
            mel = mel_spectrogram.unsqueeze(0).clone() # [1, n_mels, T]
        else:
            mel = mel_spectrogram.clone() # [B, n_mels, T]

        batch_size, n_mels, time_steps = mel.shape

        for b in range(batch_size):
            # 1. Frequency masking
            if self.freq_mask_param > 0 and n_mels > 1:
                f = random.randint(0, min(self.freq_mask_param, n_mels))
                f0 = random.randint(0, n_mels - f)
                mel[b, f0 : f0 + f, :] = self.fill_value

            # 2. Time masking
            if self.time_mask_param > 0 and time_steps > 1:
                t = random.randint(0, min(self.time_mask_param, time_steps))
                t0 = random.randint(0, time_steps - t)
                mel[b, :, t0 : t0 + t] = self.fill_value

        if was_2d:
            return mel.squeeze(0)
        return mel


class TriParallelDataset(Dataset):
    """
    PyTorch Dataset supporting Tri-Parallel (audio_vi, text_vi, text_hmong)
    and Quad-Parallel (audio_vi, text_vi, text_hmong, audio_hmong) tuples.
    Features on-the-fly audio loading, RMS normalization, and Mel-spectrogram extraction.
    """
    def __init__(
        self,
        records: Union[List[Union[Tuple, Dict[str, Any]]], str],
        sample_rate: int = 16000,
        augment: bool = False,
        extract_mel: bool = True,
        preprocessor: Optional[AudioPreprocessor] = None,
        snr_range: Tuple[float, float] = (10.0, 20.0),
        freq_mask_param: int = 15,
        time_mask_param: int = 35
    ):
        self.sample_rate = sample_rate
        self.augment = augment
        self.extract_mel = extract_mel
        self.preprocessor = preprocessor or AudioPreprocessor(sample_rate=sample_rate)

        # Augmentation components
        self.noise_augmenter = AdditiveGaussianNoise(snr_db_range=snr_range)
        self.spec_augmenter = SpecAugment(
            time_masking=time_mask_param,
            freq_masking=freq_mask_param
        )

        # Load records from list or file
        if isinstance(records, str):
            self.records = self._load_from_file(records)
        else:
            self.records = self._standardize_records(records)

    def _load_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parses JSON, TSV, or CSV datasets."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        records = []
        if file_path.endswith(".json") or file_path.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        elif file_path.endswith(".tsv"):
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                records = list(reader)
        elif file_path.endswith(".csv"):
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                records = list(reader)
        else:
            raise ValueError(f"Unsupported file format for dataset: {file_path}")

        return self._standardize_records(records)

    def _standardize_records(
        self,
        records: List[Union[Tuple, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Converts heterogeneous tuples or dicts into standard dict format."""
        standardized = []
        for item in records:
            if isinstance(item, tuple) or isinstance(item, list):
                if len(item) == 3:
                    # Tri-parallel: audio_vi, text_vi, text_hmong
                    standardized.append({
                        "audio_vi": item[0],
                        "text_vi": str(item[1]),
                        "text_hmong": str(item[2]),
                        "audio_hmong": None
                    })
                elif len(item) >= 4:
                    # Quad-parallel: audio_vi, text_vi, text_hmong, audio_hmong
                    standardized.append({
                        "audio_vi": item[0],
                        "text_vi": str(item[1]),
                        "text_hmong": str(item[2]),
                        "audio_hmong": item[3]
                    })
                else:
                    raise ValueError(f"Tuple must have at least 3 elements, got {len(item)}")
            elif isinstance(item, dict):
                standardized.append({
                    "audio_vi": item.get("audio_vi"),
                    "text_vi": str(item.get("text_vi", "")),
                    "text_hmong": str(item.get("text_hmong", "")),
                    "audio_hmong": item.get("audio_hmong", None),
                    "metadata": item.get("metadata", {})
                })
            else:
                raise TypeError(f"Invalid record type: {type(item)}")
        return standardized

    def __len__(self) -> int:
        return len(self.records)

    def load_audio(
        self,
        audio_data: Union[str, torch.Tensor, np.ndarray, List[float]]
    ) -> torch.Tensor:
        """
        Loads audio on-the-fly, converts to 16kHz mono, and applies RMS normalization.
        Returns: 1D torch.Tensor float32 of shape [N].
        """
        if isinstance(audio_data, str):
            waveform, sr = AudioPreprocessor.load_audio_file(audio_data)
            mono_16k = self.preprocessor.resample_to_16k(waveform, sr)
        elif isinstance(audio_data, (torch.Tensor, np.ndarray, list)):
            mono_16k = self.preprocessor.resample_to_16k(audio_data, self.sample_rate)
        else:
            raise TypeError(f"Unsupported audio data type: {type(audio_data)}")

        norm_wav = self.preprocessor.normalize_rms(mono_16k, target_db=-20.0)
        return norm_wav.squeeze(0) # [N]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves record at idx, loads audio on-the-fly, applies augmentations,
        and computes Mel-spectrograms.
        """
        record = self.records[idx]

        # 1. Process Vietnamese Audio
        wav_vi = self.load_audio(record["audio_vi"])
        if self.augment and self.noise_augmenter.training:
            wav_vi = self.noise_augmenter(wav_vi)

        item: Dict[str, Any] = {
            "audio_vi": wav_vi,
            "audio_vi_len": wav_vi.size(0),
            "text_vi": record["text_vi"],
            "text_hmong": record["text_hmong"]
        }

        # 2. Extract Mel-Spectrogram for Vietnamese Audio
        if self.extract_mel:
            mel_vi = self.preprocessor.extract_mel_spectrogram(wav_vi.unsqueeze(0)).squeeze(0) # [n_mels, T]
            if self.augment and self.spec_augmenter.training:
                mel_vi = self.spec_augmenter(mel_vi)
            item["mel_vi"] = mel_vi
            item["mel_vi_len"] = mel_vi.size(-1)

        # 3. Process Hmong Audio (if present in quad-parallel record)
        if record.get("audio_hmong") is not None:
            wav_hm = self.load_audio(record["audio_hmong"])
            item["audio_hmong"] = wav_hm
            item["audio_hmong_len"] = wav_hm.size(0)

            if self.extract_mel:
                mel_hm = self.preprocessor.extract_mel_spectrogram(wav_hm.unsqueeze(0)).squeeze(0)
                item["mel_hmong"] = mel_hm
                item["mel_hmong_len"] = mel_hm.size(-1)

        return item


class DynamicPaddingCollate:
    """
    Collate function performing dynamic batching and dynamic padding
    for audio waveforms, log Mel-spectrograms, and token sequences,
    eliminating redundant zero-padding across variable-length utterances.
    """
    def __init__(
        self,
        pad_value: float = 0.0,
        mel_pad_value: float = -11.5129,
        token_pad_id: int = 0,
        sort_by_length: bool = True
    ):
        self.pad_value = pad_value
        self.mel_pad_value = mel_pad_value
        self.token_pad_id = token_pad_id
        self.sort_by_length = sort_by_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collates a list of dataset item dicts into padded batch tensors.
        """
        if not batch:
            return {}

        # Optionally sort batch descending by audio_vi length for optimal packing
        if self.sort_by_length:
            batch = sorted(batch, key=lambda x: x["audio_vi_len"], reverse=True)

        batch_size = len(batch)

        # 1. Pad Audio Waveforms (Vietnamese)
        max_wav_vi_len = max(x["audio_vi_len"] for x in batch)
        padded_wav_vi = torch.full((batch_size, max_wav_vi_len), self.pad_value, dtype=torch.float32)
        wav_vi_lens = torch.tensor([x["audio_vi_len"] for x in batch], dtype=torch.long)

        for i, item in enumerate(batch):
            length = item["audio_vi_len"]
            padded_wav_vi[i, :length] = item["audio_vi"]

        collated: Dict[str, Any] = {
            "audio_vi": padded_wav_vi,
            "audio_vi_lens": wav_vi_lens,
            "text_vi": [x["text_vi"] for x in batch],
            "text_hmong": [x["text_hmong"] for x in batch]
        }

        # 2. Pad Mel-Spectrograms (Vietnamese)
        if "mel_vi" in batch[0]:
            n_mels = batch[0]["mel_vi"].size(0)
            max_mel_vi_len = max(x["mel_vi_len"] for x in batch)
            padded_mel_vi = torch.full(
                (batch_size, n_mels, max_mel_vi_len),
                self.mel_pad_value,
                dtype=torch.float32
            )
            mel_vi_lens = torch.tensor([x["mel_vi_len"] for x in batch], dtype=torch.long)

            for i, item in enumerate(batch):
                t_len = item["mel_vi_len"]
                padded_mel_vi[i, :, :t_len] = item["mel_vi"]

            collated["mel_vi"] = padded_mel_vi
            collated["mel_vi_lens"] = mel_vi_lens

        # 3. Pad Hmong Audio Waveforms (if quad-parallel)
        if "audio_hmong" in batch[0]:
            max_wav_hm_len = max(x["audio_hmong_len"] for x in batch)
            padded_wav_hm = torch.full((batch_size, max_wav_hm_len), self.pad_value, dtype=torch.float32)
            wav_hm_lens = torch.tensor([x["audio_hmong_len"] for x in batch], dtype=torch.long)

            for i, item in enumerate(batch):
                length = item["audio_hmong_len"]
                padded_wav_hm[i, :length] = item["audio_hmong"]

            collated["audio_hmong"] = padded_wav_hm
            collated["audio_hmong_lens"] = wav_hm_lens

        # 4. Pad Hmong Mel-Spectrograms (if quad-parallel)
        if "mel_hmong" in batch[0]:
            n_mels = batch[0]["mel_hmong"].size(0)
            max_mel_hm_len = max(x["mel_hmong_len"] for x in batch)
            padded_mel_hm = torch.full(
                (batch_size, n_mels, max_mel_hm_len),
                self.mel_pad_value,
                dtype=torch.float32
            )
            mel_hm_lens = torch.tensor([x["mel_hmong_len"] for x in batch], dtype=torch.long)

            for i, item in enumerate(batch):
                t_len = item["mel_hmong_len"]
                padded_mel_hm[i, :, :t_len] = item["mel_hmong"]

            collated["mel_hmong"] = padded_mel_hm
            collated["mel_hmong_lens"] = mel_hm_lens

        # 5. Token sequence dynamic padding (if token sequences are provided)
        if "token_ids_vi" in batch[0]:
            max_tokens_vi = max(len(x["token_ids_vi"]) for x in batch)
            padded_tokens_vi = torch.full((batch_size, max_tokens_vi), self.token_pad_id, dtype=torch.long)
            for i, item in enumerate(batch):
                t_len = len(item["token_ids_vi"])
                padded_tokens_vi[i, :t_len] = torch.tensor(item["token_ids_vi"], dtype=torch.long)
            collated["token_ids_vi"] = padded_tokens_vi
            collated["token_ids_vi_lens"] = torch.tensor([len(x["token_ids_vi"]) for x in batch], dtype=torch.long)

        return collated
