from generateData import *
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(f"Default device set to {device}!")


class BasicClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.act_fn = nn.ReLU()
        self.linear_layer1 = nn.Linear(in_features=2, out_features=8*8, bias=True, dtype=torch.float64)
        self.linear_layer2 = nn.Linear(in_features=8*8, out_features=8*8*8, bias=True, dtype=torch.float64)
        self.linear_layer3 = nn.Linear(in_features=8*8*8, out_features=1, bias=True, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #return self.linear_layer3(self.linear_layer2(self.linear_layer1(x)))
        out_1 = self.act_fn(self.linear_layer1(x))
        out_2 = self.act_fn(self.linear_layer2(out_1))
        out_3 = self.linear_layer3(out_2)
        return out_3