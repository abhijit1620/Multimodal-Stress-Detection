import torch
import torch.nn as nn
import torch.nn.functional as F


class SensorCNN(nn.Module):
    def __init__(self, in_channels=8, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, hidden_dim, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x: [B, C, T]
        h = self.net(x).squeeze(-1)  # [B, hidden_dim]
        return h


class FaceCNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(64, hidden_dim)

    def forward(self, x):
        h = self.cnn(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class AudioCNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(32, hidden_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        h = self.cnn(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class FusionModel(nn.Module):
    def __init__(self, num_classes=2, sensor_in_channels=8):
        super().__init__()

        self.sensor_encoder = SensorCNN(in_channels=sensor_in_channels, hidden_dim=64)
        self.face_encoder = FaceCNN(hidden_dim=64)
        self.audio_encoder = AudioCNN(hidden_dim=64)

        self.classifier = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, face=None, audio=None, sensors=None):
        if sensors is None:
            raise ValueError("Sensors tensor required")

        # ensure batch dimension
        if sensors.dim() == 2:
            sensors = sensors.unsqueeze(0)

        # if [B, T, C] -> [B, C, T]
        if sensors.shape[1] > sensors.shape[2]:
            sensors = sensors.permute(0, 2, 1)

        # ✅ DEAP fix: convert 32 channels -> 8 averaged channels
        if sensors.shape[1] == 32:
            sensors = sensors.view(sensors.size(0), 8, 4, -1).mean(dim=2)

        # ---- Sensor Encoding ----
        s = self.sensor_encoder(sensors)  # [B, 64]

        B = sensors.shape[0]

        # ---- Face Encoding ----
        if face is not None and isinstance(face, torch.Tensor):
            f = self.face_encoder(face)
        else:
            f = torch.zeros((B, 64))

        # ---- Audio Encoding ----
        if audio is not None and isinstance(audio, torch.Tensor):
            a = self.audio_encoder(audio)
        else:
            a = torch.zeros((B, 64))

        # ---- Fusion ----
        z = torch.cat([f, a, s], dim=1)  # [B, 192]
        return self.classifier(z)