"""Model is split in two on purpose.
Backbone = conv part, gets compressed and stays frozen.
Head = one linear layer, this is the part that gets personalized on the phone.
"""
import torch
import torch.nn as nn

EMB, N_RR, N_CLS = 64, 4, 5


def block(ci, co, k):
    return nn.Sequential(nn.Conv1d(ci, co, k, padding=k // 2), nn.BatchNorm1d(co),
                         nn.ReLU(), nn.MaxPool1d(2))


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(block(1, 16, 7), block(16, 32, 5), block(32, 64, 5),
                                 block(64, EMB, 3), nn.AdaptiveAvgPool1d(1), nn.Flatten())

    def forward(self, x):          # x: (B, 256)
        return self.net(x.unsqueeze(1))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(EMB + N_RR, N_CLS)

    def forward(self, z):          # z is (B, 68), caller does the concat (same as in swift)
        return self.fc(z)


class BeatNet(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.backbone, self.head = Backbone(), Head()
        self.drop = nn.Dropout(dropout)  # only on while training, no weights so old checkpoints still load

    def forward(self, x, rr):
        return self.head(torch.cat([self.drop(self.backbone(x)), rr], 1))


if __name__ == "__main__":
    m = BeatNet()
    print(sum(p.numel() for p in m.parameters()), "params")
    print(m(torch.randn(8, 256), torch.randn(8, 4)).shape)
