from src.dataset import AudioDataset
from tqdm import tqdm

# --- Training & Build Database ---

encoder = AudioEncoder().cuda()  # Instantiate the audio encoder and move it to the GPU
opt = torch.optim.Adam(encoder.parameters(), 1e-3)  # Adam optimizer for training the encoder
database = {}  # Dictionary to store embeddings for each audio clip

dataset = AudioDataset(folder_path = "../../Music/bi_test")

# clips = [wav, wav]  # Duplicate for training; in practice, use multiple different clips

# Self-supervised training loop (contrastive learning)
for epoch in tqdm(range(200)):
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





