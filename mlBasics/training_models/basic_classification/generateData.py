import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
