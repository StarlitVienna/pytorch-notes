import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(f"Default device set to {device}!")

torch.manual_seed(69)

matrix = torch.randn([2, 3])
print(matrix, "\n")
transpose_view = matrix.T
print(transpose_view)
