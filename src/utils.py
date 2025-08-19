import os
import torch
import numpy as np
import soundfile as sf
from scipy.io import wavfile


def save_wav(path, wav, sr):
    # Create the parent directory if it doesn't exist
    parent_dir = os.path.dirname(path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    
    # Convert tensor to numpy array
    if isinstance(wav, torch.Tensor):
        wav_np = wav.detach().cpu().numpy()
    else:
        wav_np = wav
    
    # Handle different shapes - soundfile expects (samples, channels) or (samples,) for mono
    if wav_np.ndim == 1:
        # Already mono (samples,)
        audio_data = wav_np
    elif wav_np.ndim == 2:
        if wav_np.shape[0] == 1:
            # Mono with channel dimension (1, samples) -> squeeze to (samples,)
            audio_data = wav_np.squeeze(0)
        elif wav_np.shape[0] == 2:
            # Stereo (2, samples) -> transpose to (samples, 2)
            audio_data = wav_np.transpose()
        else:
            # Multi-channel (channels, samples) -> transpose to (samples, channels)
            audio_data = wav_np.transpose()
    else:
        raise ValueError(f"Cannot handle array with shape {wav_np.shape}")
    
    try:
        # Use soundfile - it's more reliable than torchaudio.save
        sf.write(path, audio_data, sr)
        print(f"Successfully saved {path} with shape {audio_data.shape} at {sr}Hz")
    except Exception as e:
        print(f"Soundfile failed, trying scipy.io.wavfile...")
        try:
            # Fallback to scipy - needs int16 format
            if audio_data.dtype != np.int16:
                # Convert float to int16
                if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                    audio_data_int = (audio_data * 32767).astype(np.int16)
                else:
                    audio_data_int = audio_data.astype(np.int16)
            else:
                audio_data_int = audio_data
            
            wavfile.write(path, sr, audio_data_int)
            print(f"Successfully saved {path} with scipy.io.wavfile")
        except Exception as e2:
            print(f"Both soundfile and scipy failed!")
            print(f"Soundfile error: {e}")
            print(f"Scipy error: {e2}")
            print(f"wav shape: {wav_np.shape}, dtype: {wav_np.dtype}, sr: {sr}")
            raise