import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(f"Current device --> {device}")

step_size = 0.1
range1_2_10 = torch.arange(start=1, end=10+step_size, step=step_size)
print(range1_2_10)
print(range1_2_10.dtype) # Important, it's dtype is int64
