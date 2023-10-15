import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(f"Default torch device set to {device}!")

step_size = 1
x = torch.arange(start=1, end=10+step_size, step=step_size)
print(x)
x_reshaped = torch.reshape(x, (5, 2))
print(x_reshaped, "\n\n")

print(x_reshaped[:, 0])
print(x_reshaped[:, 1], "\n")
print(x_reshaped[0, :])