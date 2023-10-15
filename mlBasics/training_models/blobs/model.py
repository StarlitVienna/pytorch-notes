from generateData import *
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(f"Default device set to {device}!")


class BlobSolvingNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.act_fn = nn.ReLU()
        self.layer1 = nn.Linear(in_features=2, out_features=8*8, dtype=torch.float32)
        self.layer2 = nn.Linear(in_features=8*8, out_features=8*8*8, dtype=torch.float32)
        self.layer3 = nn.Linear(in_features=8*8*8, out_features=4, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_1 = self.act_fn(self.layer1(x))
        out_2 = self.act_fn(self.layer2(out_1))
        out_3 = self.layer3(out_2)

        return out_3