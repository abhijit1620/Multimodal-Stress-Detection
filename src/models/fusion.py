import torch
import torch.nn as nn
from src.models.sensor_1dcnn import Sensor1DCNN

class FusionNet(nn.Module):
    def __init__(self, num_classes=3, sensor_in_channels=6):
        super().__init__()
        # TODO: Replace with real encoders later
        self.face = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3*224*224, 512),   # 🔥 Update for 224×224
        nn.ReLU(),
        nn.Dropout(0.3)
        )          
        self.audio = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1*128*128, 128),   # dummy audio encoder -> 128 features
            nn.ReLU()
        )
        self.sensors = Sensor1DCNN(in_channels=sensor_in_channels, hidden=32)

        # final fusion layer
        self.fc = nn.Linear(512 + 128 + 32, num_classes)

    def forward(self, face, audio, sensors):
        f = self.face(face)
        a = self.audio(audio)
        s = self.sensors(sensors)
        x = torch.cat([f, a, s], dim=1)
        return self.fc(x)
