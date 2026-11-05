import sys
sys.path.insert(0, ".")
import os
import glob
import soundfile as sf
import torch
import torchaudio.functional as F

def prepare_samples():
    yt_dir = os.path.abspath('tests/data/yt_speech')
    os.makedirs(yt_dir, exist_ok=True)

    intro_path = os.path.join(yt_dir, 'intro.wav')
    raw_path = os.path.join(yt_dir, 'raw_chunk.wav')

    if not os.path.exists(intro_path) or not os.path.exists(raw_path):
        print("Source files not found.")
        return

    intro_data, intro_sr = sf.read(intro_path)
    if intro_data.ndim > 1:
        intro_data = intro_data.mean(axis=1)

    raw_data, raw_sr = sf.read(raw_path)
    if raw_data.ndim > 1:
        raw_data = raw_data.mean(axis=1)

    # 1. yt_sample_01.wav: 0.0s to 5.5s of intro (News broadcast intro & patrol)
    s1 = intro_data[int(0.0 * intro_sr):int(5.5 * intro_sr)]
    s1_16k = F.resample(torch.from_numpy(s1).float().unsqueeze(0), intro_sr, 16000).squeeze(0).numpy()
    p1 = os.path.join(yt_dir, 'yt_sample_01.wav')
    sf.write(p1, s1_16k, 16000, subtype='PCM_16')

    # 2. yt_sample_02.wav: 6.0s to 9.8s of intro (Law enforcement report)
    s2 = intro_data[int(6.0 * intro_sr):int(9.8 * intro_sr)]
    s2_16k = F.resample(torch.from_numpy(s2).float().unsqueeze(0), intro_sr, 16000).squeeze(0).numpy()
    p2 = os.path.join(yt_dir, 'yt_sample_02.wav')
    sf.write(p2, s2_16k, 16000, subtype='PCM_16')

    # 3. yt_sample_03.wav: 40.0s to 45.0s of raw_chunk (Hospital & medical care report)
    s3 = raw_data[int(40.0 * raw_sr):int(45.0 * raw_sr)]
    s3_16k = F.resample(torch.from_numpy(s3).float().unsqueeze(0), raw_sr, 16000).squeeze(0).numpy()
    p3 = os.path.join(yt_dir, 'yt_sample_03.wav')
    sf.write(p3, s3_16k, 16000, subtype='PCM_16')

    print("Saved 3 target WAV files:")
    for p in [p1, p2, p3]:
        info = sf.info(p)
        print(f"  {os.path.basename(p)}: duration={info.duration:.2f}s, sr={info.samplerate}, channels={info.channels}, subtype={info.subtype}")

    # Clean up intermediate large raw chunks
    for temp_f in [intro_path, raw_path, os.path.join(yt_dir, 'chunk_01.wav'), os.path.join(yt_dir, 'temp.wav'), os.path.join(yt_dir, 'test_scan.wav'), os.path.join(yt_dir, 'temp_cand.wav'), os.path.join(yt_dir, 'test_cand.wav')]:
        if os.path.exists(temp_f):
            try:
                os.remove(temp_f)
                print(f"Cleaned up {os.path.basename(temp_f)}")
            except Exception as e:
                print(f"Could not remove {temp_f}: {e}")

if __name__ == '__main__':
    prepare_samples()
