import numpy as np
import torch
import torch.nn as nn

X = np.load("X.npy", allow_pickle=True)
y = np.load("y.npy")

max_len = 30
feature_size = len(X[0][0])

X_padded = np.zeros((len(X), max_len, feature_size))

for i, seq in enumerate(X):
    for j in range(min(len(seq), max_len)):
        X_padded[i][j] = seq[j]

X_tensor = torch.tensor(X_padded).float()
y_tensor = torch.tensor(y).long()

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(feature_size, 128, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

model = LSTMModel()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    output = model(X_tensor)
    loss = criterion(output, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), "anomaly_model.pth")

print("✅ Model trained and saved")