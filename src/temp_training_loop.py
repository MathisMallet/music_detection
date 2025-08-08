for epoch in range(num_epochs):
    encoder.train()
    
    for wav_batch, sr in dataloader:  # wav_batch shape: (n, time)
        all_embeddings = []
        all_labels = []

        # Generate p augmentations for each sample
        for i in range(p):
            augmented_batch = [augment_audio(w, sr) for w in wav_batch]   # list of len n
            mel_batch = [get_mel_spec(w, sr) for w in augmented_batch]    # list of len n
            mel_batch = torch.stack(mel_batch).cuda()  # shape: (n, mel_bins, time_frames)
            z_batch = encoder(mel_batch)               # shape: (n, embedding_dim)

            all_embeddings.append(z_batch)
            all_labels.append(torch.arange(len(wav_batch)))  # labels: 0..n-1

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
