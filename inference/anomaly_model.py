import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):

    def __init__(self, input_dim=133, hidden_dim=128):  # ✅ FIX HERE
        super().__init__()

        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)

        loss = ((x - decoded) ** 2).mean(dim=(1, 2))
        return loss