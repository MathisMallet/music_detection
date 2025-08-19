import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import librosa
import random
import numpy as np
import faiss
import gradio as gr
import os
import soundfile as sf
from scipy.io import wavfile

# --- Utils ---
def to_mono(wav):
    if wav.dim() == 1:
        # Already mono, just add channel dimension
        return wav.unsqueeze(0)
    elif wav.dim() == 2:
        if wav.shape[0] == 1:
            # Already mono with channel dimension
            return wav
        else:
            # Multi-channel, convert to mono by averaging channels
            return wav.mean(dim=0, keepdim=True)
    else:
        raise ValueError(f"Unexpected tensor dimensions: {wav.shape}")

def mono_to_stereo(wav):
    """
    Convert mono audio to stereo (dual channel)
    
    Args:
        wav: Mono audio tensor of shape (1, samples) or (samples,)
    
    Returns:
        Stereo audio tensor of shape (2, samples)
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # Add channel dimension
    
    if wav.shape[0] != 1:
        raise ValueError(f"Expected mono audio (1 channel), got {wav.shape[0]} channels")
    
    mono_signal = wav[0]  # Extract the mono signal (samples,)

    # Simple duplication: same signal in both left and right channels
    stereo = torch.stack([mono_signal, mono_signal], dim=0)
    
    return stereo

# --- Data Augmentations ---
def augment_audio(wav, sr):
    arr = wav.numpy().squeeze()
    arr = librosa.effects.pitch_shift(y=arr, sr=sr, n_steps=random.choice([-2, -1, 0, 1, 2]))
    arr = librosa.effects.time_stretch(y=arr, rate=random.uniform(0.9, 1.1))
    return torch.tensor(arr).unsqueeze(0)

def get_mel_spec(wav, sr):
    spec = T.MelSpectrogram(sr, n_mels=64)(wav)
    return T.AmplitudeToDB()(spec)

# --- Encoder ---
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    def forward(self, x):
        return self.net(x).view(x.size(0), -1)

def contrastive_loss(z1, z2, temp=0.5):
    z1 = nn.functional.normalize(z1, 1)
    z2 = nn.functional.normalize(z2, 1)
    reps = torch.cat([z1, z2], 0)
    sim = reps @ reps.T
    mask = torch.eye(len(reps), device=sim.device).bool()
    sim = sim[~mask].view(len(reps), -1)
    pos = torch.exp(torch.sum(z1 * z2, 1) / temp)
    pos = torch.cat([pos, pos], 0)
    denom = torch.sum(torch.exp(sim / temp), 1)
    return (-torch.log(pos / denom)).mean()

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

# --- Training & Build Database ---

encoder = AudioEncoder().cuda()  # Instantiate the audio encoder and move it to the GPU
opt = torch.optim.Adam(encoder.parameters(), 1e-3)  # Adam optimizer for training the encoder
database = {}  # Dictionary to store embeddings for each audio clip

# Load demo clip (replace or expand as needed)
url = "tutorial-assets/steam-train-whistle-daniel_simon.wav"
wav, sr = torchaudio.load(torchaudio.utils.download_asset(url))  # Download and load the audio file
wav = to_mono(wav)  # Convert to mono channel
save_wav("tmp/raw_audio_wav/processed_wav.wav", wav, sr) #save the audio to a wav file
clips = [wav, wav]  # Duplicate for training; in practice, use multiple different clips

# Self-supervised training loop (contrastive learning)
for epoch in range(3):
    encoder.train()  # Set encoder to training mode
    aug1, aug2 = augment_audio(wav, sr), augment_audio(wav, sr)  # Two random augmentations of the same audio
    save_wav("tmp/raw_audio_wav/processed_wav1.wav", aug1, sr) #save the audio to a wav file
    save_wav("tmp/raw_audio_wav/processed_wav2.wav", aug2, sr) #save the audio to a wav file
    m1, m2 = get_mel_spec(aug1, sr), get_mel_spec(aug2, sr)  # Convert both to mel spectrograms
    z1 = encoder(m1.unsqueeze(0).cuda())  # Get embedding for first augmentation
    z2 = encoder(m2.unsqueeze(0).cuda())  # Get embedding for second augmentation
    loss = contrastive_loss(z1, z2)  # Compute contrastive loss between the two embeddings
    opt.zero_grad()  # Clear gradients
    loss.backward()  # Backpropagate
    opt.step()  # Update encoder weights
    print("Epoch", epoch, "Loss", loss.item())  # Print training loss

# --- Build Embedding Database ---
encoder.eval()  # Set encoder to evaluation mode
embs = []  # List to store embeddings
labels = []  # List to store corresponding labels

# For each clip (here, just one), compute and store its embedding
for name, clip in {"steam": wav}.items():
    mono_clip = to_mono(clip)  # Ensure mono
    m = get_mel_spec(mono_clip, sr).unsqueeze(0).cuda()  # Mel spectrogram, batch dimension, move to GPU
    with torch.no_grad():
        em = encoder(m).cpu().numpy()[0]  # Get embedding, move to CPU, convert to numpy
    database[name] = em  # Store in database dictionary
    embs.append(em)  # Add to embedding list
    labels.append(name)  # Add label

dim = embs[0].shape[0]  # Dimensionality of embeddings
index = faiss.IndexFlatL2(dim)  # Create a FAISS index for fast nearest neighbor search (L2 distance)
index.add(np.array(embs))  # Add all embeddings to the index

# --- Gradio Demo ---
def recognize(inp):
    if inp is None:
        return "Please upload an audio file"
    
    try:
        wav2, sr2 = torchaudio.load(inp)
        wav2 = to_mono(wav2)
        aug = augment_audio(wav2, sr2)
        m = get_mel_spec(aug, sr2).unsqueeze(0).cuda()
        with torch.no_grad():
            em2 = encoder(m).cpu().numpy()
        D, I = index.search(em2, 1)
        return f"Closest Match: {labels[I[0][0]]} (dist {D[0][0]:.3f})"
    except Exception as e:
        return f"Error processing audio: {str(e)}"

gr.Interface(
    fn=recognize,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Music Recognizer Demo"
).launch()
