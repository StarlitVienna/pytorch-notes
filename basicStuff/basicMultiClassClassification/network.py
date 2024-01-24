import torch
from torch import nn


class MultiClassModel(nn.Module):
    def __init__(self, device, n_features, n_classes):
        super().__init__()
        self.device = device


        self.layer1 = nn.Linear(in_features=n_features, out_features=8*8, bias=True, device=device, dtype=torch.float32)
        self.layer2 = nn.Linear(in_features=8*8, out_features=8*8*8, bias=True, device=device, dtype=torch.float32)
        self.layer3 = nn.Linear(in_features=8*8*8, out_features=8*8*8*8, bias=True, device=device, dtype=torch.float32)
        self.layer4 = nn.Linear(in_features=8*8*8*8, out_features=n_classes, bias=True, device=device, dtype=torch.float32)

        self.act_fun = nn.ReLU()

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        out1 = self.layer1(X)
        out2 = self.layer2(self.act_fun(out1))
        out3 = self.layer3(self.act_fun(out2))
        out4 = self.layer4(self.act_fun(out3))

        return out4

