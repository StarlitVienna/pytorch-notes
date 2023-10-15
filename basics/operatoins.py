import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(f"Default device set to {device}!")


tensor1 = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
tensor2 = torch.tensor([5, 6, 7, 8, 9, 10], dtype=torch.int64)
mult_result = torch.mul(tensor1, tensor2)
# or
mult_result = tensor1 * tensor2
print(mult_result)



#Mult by scalar
scalar_mult_result = torch.mul(tensor1, 5)
# or
scalar_mult_result = tensor1 * 5
print(scalar_mult_result)



#Cast to all (addition)
cast_result = tensor1 + 10
print(cast_result)