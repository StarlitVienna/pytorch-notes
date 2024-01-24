import torch
from torch import nn

class BasicClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Linear(in_features=2, out_features=8*8, bias=True, dtype=torch.float64)
        self.layer2 = nn.Linear(in_features=8*8, out_features=8*8*8, bias=True, dtype=torch.float64)
        self.layer3 = nn.Linear(in_features=8*8*8, out_features=1, bias=True, dtype=torch.float64)

        self.act_func = nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layer3(self.act_func(self.layer2(self.act_func(self.layer1(X)))))
