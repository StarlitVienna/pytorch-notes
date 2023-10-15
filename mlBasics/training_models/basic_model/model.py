from datasetGenerator import * #No need to import torch because of this
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(f"Default device set to {device}!")


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn((1), requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn((1), requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.weights * x + self.bias)


#The loss function might be reffered as cost function of criterion
def training_model():
    pass
