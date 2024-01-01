from generateData import *
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(f"Torch default device set to {device}!")

class CircleDataModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer1 = nn.Linear(in_features=2, out_features=8*8, dtype=torch.float64)
        self.linear_layer2 = nn.Linear(in_features=8*8, out_features=8*8*8, dtype=torch.float64)
        self.linear_layer3 = nn.Linear(in_features=8*8*8, out_features=2, dtype=torch.float64)
        self.act_fn = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.act_fn(self.linear_layer1(x))
        out2 = self.act_fn(self.linear_layer2(out1))
        out3 = self.linear_layer3(out2)
        return out3