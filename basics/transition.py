import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(f"Default device set to {device}!")

x = torch.arange(1, 10+1, step=1)

#Torch tensor to numpy
x_numpy = x.cpu().numpy() # numpy cannot work with the gpu, thus the reason to copy it to the cpu
print(x_numpy)

#Numpy to torch tensor
x_numpy_2_tensor = torch.from_numpy(x_numpy)
print(x_numpy_2_tensor)
print(x_numpy_2_tensor.dtype)
print(x_numpy_2_tensor.device) # its device will revert back to cpu
x_numpy_2_tensor = x_numpy_2_tensor.to(device)
print(x_numpy_2_tensor.device)
