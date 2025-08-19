from src.dataset import AudioDataset
from tqdm import tqdm
from src.augmenter import augment_audio
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import faiss
import gradio as gr
import torchaudio.transforms as T

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

encoder = AudioEncoder().cuda()  # Instantiate the audio encoder and move it to the GPU
opt = torch.optim.Adam(encoder.parameters(), 1e-3)  # Adam optimizer for training the encoder
database = {}  # Dictionary to store embeddings for each audio clip

dataset = AudioDataset(folder_path = "../../Music/bi_test")
num_epochs = 20
p = 30

batch_size = 2
cpt = 0
all_batch = []
wav_batch = []
for wav, sr, label in dataset:
    if cpt < batch_size :
        wav_batch.append(wav)
        cpt += 1
    elif cpt == batch_size:
        all_batch.append(wav_batch)
        wav_batch = [wav]
        cpt = 1
    else :
        raise Exception("should not happen")
if len(wav_batch) > 1 :
    all_batch.append(wav_batch)
    


for epoch in range(num_epochs):
    encoder.train()
    
    for wav_batch in all_batch: 
        all_embeddings = []
        all_labels = []

        # Generate p augmentations for each sample
        for i in tqdm(range(p)):
            augmented_batch = [augment_audio(w, sr) for w in wav_batch]   # list of len n
            mel_batch = [get_mel_spec(w, sr) for w in augmented_batch]    # list of len n
            mel_batch = torch.stack(mel_batch).cuda()  # shape: (n, mel_bins, time_frames)
            z_batch = encoder(mel_batch)               # shape: (n, embedding_dim)

            all_embeddings.append(z_batch)
            all_labels.append(torch.arange(len(wav_batch)))  # labels: 0..n-1
            print(all_labels)

        # Concatenate all p augmentations along the batch dimension
        all_embeddings = torch.cat(all_embeddings, dim=0)  # shape: (n*p, embedding_dim)
        all_labels = torch.cat(all_labels, dim=0).cuda()    # shape: (n*p,)

        # Compute contrastive loss (e.g., NT-Xent)
        loss = contrastive_loss(all_embeddings, all_labels)

        # Backprop
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")


# --- Build Embedding Database ---
encoder.eval()  # Set encoder to evaluation mode
embs = []  # List to store embeddings
labels = []  # List to store corresponding labels


# For each clip, compute and store its embedding
for wav, sr, label in dataset:
    m = get_mel_spec(wav, sr).unsqueeze(0).cuda()  # Mel spectrogram, batch dimension, move to GPU
    with torch.no_grad():
        em = encoder(m).cpu().numpy()[0]  # Get embedding, move to CPU, convert to numpy
    database[label] = em  # Store in database dictionary
    embs.append(em)  # Add to embedding list
    labels.append(label)  # Add label

# for name, clip in {"steam": wav}.items():
#     m = get_mel_spec(mono_clip, sr).unsqueeze(0).cuda()  # Mel spectrogram, batch dimension, move to GPU
#     with torch.no_grad():
#         em = encoder(m).cpu().numpy()[0]  # Get embedding, move to CPU, convert to numpy
#     database[name] = em  # Store in database dictionary
#     embs.append(em)  # Add to embedding list
#     labels.append(name)  # Add label

dim = embs[0].shape[0]  # Dimensionality of embeddings
index = faiss.IndexFlatL2(dim)  # Create a FAISS index for fast nearest neighbor search (L2 distance)
index.add(np.array(embs))  # Add all embeddings to the index

# --- Gradio Demo ---
def recognize(inp):
    if inp is None:
        return "Please upload an audio file"
    
    try:
        wav2, sr2 = torchaudio.load(inp)
        if sr2 != 22050:
            wav2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=22050)(wav2)
        wav2 = torch.mean(wav2, dim=0)
        m = get_mel_spec(wav2, sr2).unsqueeze(0).cuda()
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