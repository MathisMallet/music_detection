import torch
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
from torch.utils.data import DataLoader
import os
import time

class AudioDataset(Dataset):
    def __init__(self, folder_path, target_sr=22050, normalize=True):
        self.folder_path = folder_path
        self.target_sr = target_sr
        self.file_paths = self.get_all_filepaths()
        self.normalize = normalize

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        init_time = time.time()
        print("loading file n°", idx)
        file_path = self.file_paths[idx]
        wav, sr = torchaudio.load(file_path)  # wav shape: (channels, time)
        wav = torch.mean(wav, dim=0)  # mono
        if sr != self.target_sr:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)(wav)
            sr = self.target_sr
        print(f"file loaded at sr={sr} Hz in", time.time() - init_time, "seconds")
        return wav, sr, os.path.basename(file_path)

    def get_all_filepaths(self):
        folder = Path(self.folder_path)
        file_paths = [str(p) for p in folder.rglob("*")]
        return file_paths

    def get_time_length(self, wav):
        return len(wav[0]) / self.target_sr

if __name__ == "__main__":
    # batch_size = 2
    dataset = AudioDataset(folder_path = "../../Music/bi_test")
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # data = dataloader.dataset 
    # wav1 = dataloader.dataset[1]  
    # print(data)
    # print(wav1)
    # print(dataset.get_time_length(wav1))
    # for batch_wav, batch_sr, batch_label in dataloader:
    #     print(f"Batch shape: {batch_wav.shape}, Sample rate: {batch_sr}, label: {batch_label}")

    mus1 = dataset[0]
    mus2 = dataset[1]
    print(mus2[0], len(mus2[0]))
    print(mus2[1])
    print(mus2[2])
    print(mus1[0], len(mus1[0]))
    print(mus1[1])
    print(mus1[2])