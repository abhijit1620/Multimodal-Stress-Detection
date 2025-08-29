import torch.nn as nn
from torchvision.models import resnet18

class FaceCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False)
        m.maxpool = nn.Identity()
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.proj = nn.Linear(512, out_dim)

    def forward(self,x):
        h = self.backbone(x).flatten(1)
        return self.proj(h)