import torch.nn as nn

class Sensor1DCNN(nn.Module):
    def __init__(self, in_channels=6, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x shape: [B, C, T]  (channels first)
        h = self.net(x).squeeze(-1)  # -> [B, hidden]
        return h
