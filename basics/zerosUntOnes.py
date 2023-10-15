import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(f"Set device is {device}\n\n")

x = 2
y = 2

"""
#Tensor of zeroes
zeroes = torch.zeros(size=(7,8))
print(zeroes)
print(zeroes.ndim)
print(zeroes.shape)
print(zeroes.dtype)
"""

"""
ones = torch.ones(size=(x,y))
print(ones)
"""