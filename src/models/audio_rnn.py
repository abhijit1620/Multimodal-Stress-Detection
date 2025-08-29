import torch.nn as nn

class AudioRNN(nn.Module):
    def __init__(self, in_feat=40, hidden=128, out_dim=128):
        super().__init__()
        self.rnn = nn.LSTM(in_feat, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden*2, out_dim)

    def forward(self, x):  # (B,T,F)
        h,_ = self.rnn(x)
        return self.proj(h[:,-1,:])