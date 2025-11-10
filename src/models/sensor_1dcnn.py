import torch
import torch.nn as nn

class SensorModel(nn.Module):
    def __init__(self, input_channels=8, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, hidden, kernel_size=3, stride=2),
            nn.AdaptiveAvgPool1d(1)  # -> [B, hidden, 1]
        )

    def forward(self, x):
        h = self.net(x).squeeze(-1)  # -> [B, hidden]
        return h