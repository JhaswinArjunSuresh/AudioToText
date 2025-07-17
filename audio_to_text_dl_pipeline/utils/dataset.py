from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import COMMONVOICE
import torchaudio

class SimpleDataset(Dataset):
    def __init__(self, split="train"):
        self.ds = COMMONVOICE(".", version="cv-corpus-13.0-2023-03-09", 
                              download=True, subset=split)
        self.processor = torchaudio.transforms.MelSpectrogram()

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        waveform, _, _, transcript, _, _, _ = self.ds[idx]
        features = self.processor(waveform).squeeze(0).transpose(0, 1)
        return features, transcript

def collate_fn(batch):
    x, y = zip(*batch)
    return x, y

def get_dataloader():
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    input_dim = 128
    vocab_size = 100
    return dataloader, input_dim, vocab_size
