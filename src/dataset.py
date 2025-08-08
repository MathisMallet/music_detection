import torch
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path

class AudioDataset(Dataset):
    def __init__(self, folder_path, target_sr=22050):
        self.folder_path = folder_path
        self.target_sr = target_sr
        self.file_paths = self.get_all_filepaths()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.file_paths[idx])  # wav shape: (channels, time)
        wav = torch.mean(wav, dim=0)  # mono
        if sr != self.target_sr:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)(wav)
            sr = self.target_sr
        return wav, sr

    def get_all_filepaths(self):
        folder = Path(self.folder_path)
        file_paths = [str(p) for p in folder.rglob("*")]
        return file_paths

# from torch.utils.data import DataLoader

# dataset = AudioDataset(list_of_audio_files, target_sr=16000)
# dataloader = DataLoader(dataset, batch_size=n, shuffle=True, drop_last=True)

if __name__ == "__main__":
    all_paths = get_all_filepaths("../../Music/bi_test")
    test = torchaudio.load(all_paths[0])
    len(test)