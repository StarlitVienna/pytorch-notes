from torch import nn
from generateData import *
#device = "cuda" if torch.cuda.is_available() else "cpu"
#torch.set_default_device(device)
#print(f"Default device set to {device}!")


class FashionMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        #self.linear1 = nn.Linear(in_features=)
        self.act_fn = nn.ReLU()
        self.flatten_layer = nn.Flatten()
        self.linear_layer1 = nn.Linear(in_features=28*28, out_features=8*8*8)
        self.linear_layer2 = nn.Linear(in_features=8*8*8, out_features=8*8*8*8)
        self.linear_layer3 = nn.Linear(in_features=8*8*8*8, out_features=10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flatten = self.flatten_layer(x)
        out_1 = self.act_fn(self.linear_layer1(flatten))
        out_2 = self.act_fn(self.linear_layer2(out_1))
        out_3 = self.linear_layer3(out_2)
        return out_3