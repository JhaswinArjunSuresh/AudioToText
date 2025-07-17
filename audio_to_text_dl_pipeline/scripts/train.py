import torch
from torch.utils.data import DataLoader
from models.simple_asr import SimpleASRModel
from utils.dataset import get_dataloader
from utils.train_utils import train_model

def main():
    dataloader, input_dim, vocab_size = get_dataloader()
    model = SimpleASRModel(input_dim=input_dim, vocab_size=vocab_size)
    train_model(model, dataloader)

if __name__ == "__main__":
    main()
