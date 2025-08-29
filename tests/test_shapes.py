import torch
from src.models.fusion import FusionNet

def test_fusion_forward():
    m = FusionNet(num_classes=3)
    face = torch.randn(2,3,112,112)
    audio = torch.randn(2,100,40)
    sensors = torch.randn(2,6,256)
    out = m(face,audio,sensors)
    assert out.shape == (2,3)