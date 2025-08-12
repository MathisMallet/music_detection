import numpy as np
import librosa
import torch


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
        self.vad = torchaudio.transforms.Vad(
            sample_rate=sample_rate, trigger_level=7.0)

    def __call__(self, audio_data):
        audio_length = audio_data.shape[0]
        if audio_length > self.clip_length:
            offset = random.randint(0, audio_length-self.clip_length)
            audio_data = audio_data[offset:(offset+self.clip_length)]
        else :
            raise Exception("Audio shorter than clip ...")
        return self.vad(audio_data) # remove silences at the beggining/end

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

def augment_audio(wav, sr):
    clip_transform = RandomClip(sr, clipt_length = 7)
    transformed_audio = clip_transform(wav)
    arr = wav.numpy().squeeze()
    arr = librosa.effects.pitch_shift(y=arr, sr=sr, n_steps=random.choice([-2, -1, 0, 1, 2]))
    arr = librosa.effects.time_stretch(y=arr, rate=random.uniform(0.9, 1.1))
    return torch.tensor(arr).unsqueeze(0)

