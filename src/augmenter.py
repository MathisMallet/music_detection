import numpy as np
import librosa
import torch
import random
import torchaudio
from src.dataset import AudioDataset
from src.utils import save_wav


def normalize(wav):
    # Normalize to [-1, 1]
    max_val = wav.abs().max()
    if max_val > 0:
        wav = wav / max_val
    else:
        raise Exception("Audio is empty ...")
    return wav

class RandomClip:
    def __init__(self, sample_rate, clip_length):
        self.clip_length = sample_rate * clip_length
        self.sr = sample_rate
        self.vad = torchaudio.transforms.Vad(
            sample_rate=sample_rate, trigger_level=7.0)

    def __call__(self, audio_data):
        audio_length = audio_data.shape[0]
        print(audio_length)
        print(len(audio_data))
        print(audio_data)
        print(len(audio_data))
        if audio_length > self.clip_length:
            offset = random.randint(0, audio_length-self.clip_length)
            print(f"Audio cut between {offset/self.sr}s and, {(offset+self.clip_length)/self.sr}s")
            audio_data = audio_data[offset:(offset+self.clip_length)]
        else :
            raise Exception("Audio shorter than clip ...")
        return audio_data # remove silences at the beggining/end  self.vad()

# clip_transform = RandomClip(sample_rate, 7) # 8 seconds clip
# transformed_audio = clip_transform(audio_data)



# def segment_audio(wav, sr, avg_time = 7, deviation = 3, sampl_interval = 1):
#     avg_num = avg_time * sr
#     deviation_num = deviation * sr
#     sampl_interval_num = sampl_interval * sr

#     total_len = len(wav)
#     small_strip = avg_num - deviation_num
#     big_strip = avg_num + deviation_num

#     segments = []
#     for start in range(0, total_len - small_strip, sampl_interval):
#         strip = np.random.randint(low = 0, high = deviation_num)
#         end  = star + strip
#         if end >= total_len :
#             outlier = end - (total_len-1)
#             start = start - outlier
#             end = end - outlier
#             print("outlier", outlier)
#         print("start", start, "end", end)
#         segment = wav[start, end]
#         segments.append(segment)

#     left_limit = random_start - half_strip
#     right_limit = random_start + half_strip
#     if right_limit >= total_len :
#         outlier = right_limit - (total_len-1)
#         left_limit = left_limit - outlier
#         right_limit = right_limit - outlier
#     if left_limit < 0 :
#         outlier = np.abs(left_limit)
#         left_limit = left_limit + outlier
#         right_limit = right_limit + outlier

def augment_audio(wav, sr, test = False): # TO DO implement noise adding
    if test:
        save_wav(path="tmp/test_augmenter/raw.wav", wav=wav, sr=sr)

    arr = wav.numpy().squeeze()
    n_steps=random.choice([-2, -1, 0, 1, 2])
    print(f"pitch schift of {n_steps} steps")
    arr = librosa.effects.pitch_shift(y=arr, sr=sr, n_steps=n_steps)
    if test:
        save_wav(path="tmp/test_augmenter/pitch.wav", wav=torch.tensor(arr).unsqueeze(0), sr=sr)

    
    rate = random.uniform(0.9, 1.1)
    print(f"time stretch rate of {rate}")
    arr = librosa.effects.time_stretch(y=arr, rate=rate)
    wav = torch.tensor(arr) #.unsqueeze(0)
    if test:
        save_wav(path="tmp/test_augmenter/stretch.wav", wav=wav, sr=sr)

    clip_transform = RandomClip(sr, clip_length = 7)
    len(wav)
    clipped_audio = clip_transform(wav)
    if test:
        save_wav(path="tmp/test_augmenter/clipped.wav", wav=clipped_audio, sr=sr)

    return clipped_audio.unsqueeze(0)

if __name__=="__main__":
    dataset = AudioDataset(folder_path = "../../Music/bi_test")
    test_audio = dataset[0]
    wav = augment_audio(wav=test_audio[0], sr=test_audio[1], test = True)
    len(wav)
