import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to {device}")

float32_tensor = torch.tensor([1.0, 2.0, 3.0], 
                                dtype=None,
                                device=device,
                                requires_grad=False)

print(float32_tensor.dtype)
# Default dtype is float32

#Type conversion
float16_tensor = float32_tensor.type(torch.float16)
print(float16_tensor.dtype)



#Set dtype when defining a tensor
int64_tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64)
print(int64_tensor.dtype)