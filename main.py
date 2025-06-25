import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import librosa
import random
import numpy as np
import faiss
import gradio as gr

# --- Utils ---
def to_mono(wav):
    arr = wav.numpy()
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    return torch.tensor(arr).unsqueeze(0)

# --- Data Augmentations ---
def augment_audio(wav, sr):
    arr = wav.numpy().squeeze()
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
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

# --- Training & Build Database ---
encoder = AudioEncoder().cuda()
opt = torch.optim.Adam(encoder.parameters(), 1e-3)
database = {}

# Load demo clip (replace or expand as needed)
url = "tutorial-assets/steam-train-whistle-daniel_simon.wav"
wav, sr = torchaudio.load(torchaudio.utils.download_asset(url))
wav = to_mono(wav)
clips = [wav, wav]  # duplicate for training; use multiple for real

for epoch in range(3):
    encoder.train()
    aug1, aug2 = augment_audio(wav, sr), augment_audio(wav, sr)
    m1, m2 = get_mel_spec(aug1, sr), get_mel_spec(aug2, sr)
    z1 = encoder(m1.unsqueeze(0).cuda()); z2 = encoder(m2.unsqueeze(0).cuda())
    loss = contrastive_loss(z1, z2)
    opt.zero_grad(); loss.backward(); opt.step()
    print("Epoch", epoch, "Loss", loss.item())

# --- Build Embedding Database ---
encoder.eval()
embs = []
labels = []
for name, clip in {"steam": wav}.items():
    mono_clip = to_mono(clip)
    m = get_mel_spec(mono_clip, sr).unsqueeze(0).cuda()
    with torch.no_grad():
        em = encoder(m).cpu().numpy()[0]
    database[name] = em
    embs.append(em)
    labels.append(name)

dim = embs[0].shape[0]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embs))

# --- Gradio Demo ---
def recognize(inp):
    wav2, sr2 = torchaudio.load(inp)
    wav2 = to_mono(wav2)
    aug = augment_audio(wav2, sr2)
    m = get_mel_spec(aug, sr2).unsqueeze(0).cuda()
    with torch.no_grad():
        em2 = encoder(m).cpu().numpy()
    D, I = index.search(em2, 1)
    return f"Closest Match: {labels[I[0][0]]} (dist {D[0][0]:.3f})"

gr.Interface(
    fn=recognize,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Music Recognizer Demo"
).launch()
