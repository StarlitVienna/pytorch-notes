import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(f"Torch default device se to {device}!")

#Setting a manual seed
SEED = 42

torch.manual_seed(SEED)
x = torch.randn(2, 2)
y = torch.randn(2, 2)
print(x)
print(y)