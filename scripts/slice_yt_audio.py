import sys
sys.path.insert(0, ".")
import soundfile as sf
import numpy as np
import torch
import torchaudio.functional as F
from src.models.stt_module import VietnameseSTT

def main():
    stt = VietnameseSTT(device='cpu')
    raw_data, raw_sr = sf.read('tests/data/yt_speech/chunk_01.wav')
    if raw_data.ndim > 1:
        raw_data = raw_data.mean(axis=1)

    intervals = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]
    for s, e in intervals:
        chunk = raw_data[int(s * raw_sr):int(e * raw_sr)]
        chunk_t = torch.from_numpy(chunk).float().unsqueeze(0)
        chunk_16k = F.resample(chunk_t, raw_sr, 16000).squeeze(0).numpy()
        sf.write('tests/data/yt_speech/temp.wav', chunk_16k, 16000)
        trans = stt.transcribe_file('tests/data/yt_speech/temp.wav')
        print(f"[{s:02d}s - {e:02d}s]: \"{trans}\"")

if __name__ == '__main__':
    main()
