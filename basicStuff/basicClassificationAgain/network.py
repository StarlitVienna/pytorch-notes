import torch
from torch import nn


class BasicBinaryClassificationModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        #2 In features because it's based on X and Y (2 coordinates systems)
        self.layer1 = nn.Linear(in_features=2, out_features=8*8, bias=True, device=device, dtype=torch.float64)
        self.layer2 = nn.Linear(in_features=8*8, out_features=8*8*8, bias=True, device=device, dtype=torch.float64)
        self.layer3 = nn.Linear(in_features=8*8*8, out_features=1, bias=True, device=device, dtype=torch.float64)

        self.act_func = nn.ReLU()
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out1 = self.layer1(X)
        out2 = self.layer2(self.act_func(out1))
        out3 = self.layer3(self.act_func(out2))
        return out3
        #out3 is going to get into a sigmoid funciton later on beacause of BCEWithLogitsLoss
        #thus the reason not to have the activation function on it