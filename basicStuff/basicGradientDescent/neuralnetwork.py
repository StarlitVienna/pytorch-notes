import torch
from torch import nn

# Basic model for linear regression
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float64))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float64))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X*self.weights+self.bias

