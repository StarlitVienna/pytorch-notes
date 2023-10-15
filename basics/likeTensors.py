import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(f"Default device --> {device}")

random_vector_tensor = torch.randn(size=(5, 5))
tensor_like_zeroes = torch.zeros_like(random_vector_tensor)
tensor_like_ones = torch.ones_like(random_vector_tensor)
print(tensor_like_zeroes)