import torch
import torch.nn as nn


class SpamClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(vocab_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)