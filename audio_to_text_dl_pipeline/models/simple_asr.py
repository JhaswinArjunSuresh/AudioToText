import torch.nn as nn

class SimpleASRModel(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super(SimpleASRModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GRU(32, 64, batch_first=True, bidirectional=True),
            nn.Linear(64 * 2, vocab_size)
        )

    def forward(self, x):
        x = self.network[0](x.transpose(1, 2))
        x = self.network[1](x)
        x, _ = self.network[2](x.transpose(1, 2))
        return self.network[3](x)
