from datasetGenerator import *
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(f"Default device set to {device}!")


class BasicLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        #self.weights = nn.Parameter(torch.randn((1), requires_grad=True, dtype=torch.float32))
        #self.bias = nn.Parameter(torch.randn((1), requires_grad=True, dtype=torch.float32))
        self.linear_layer = nn.Linear(in_features=1, out_features=1, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
