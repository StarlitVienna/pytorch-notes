import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(f"Current device set to --> {device}")

x = torch.tensor([1, 2, 3, 4])
y = torch.tensor([9, 10, 11, 12])



"""
#Stacking
z = torch.stack((x,y))
print(z)
print(z.ndim)
print(z.shape)
can also specify the dimension in which to stack
"""



"""
#Squeezing
w = torch.tensor([[[1, 2, 3]]])
t = torch.squeeze(w)
print(w)
print(w.ndim)
print(w.shape)
print(t)
print(t.shape)
"""



"""
#Unsqueezing
unsqueezed_on_first_dimension = torch.unsqueeze(x, 1)
unsqueezed_on_zero = torch.unsqueeze(x, 0)
print(x)
print(f"X shape --> {x.shape}")
print(unsqueezed_on_first_dimension)
print(unsqueezed_on_first_dimension.ndim)
print(unsqueezed_on_first_dimension.shape, "\n")

print(unsqueezed_on_zero)
print(unsqueezed_on_zero.shape)
"""



"""
#Reshaping
x_reshaped = x.reshape(4, 1) # same as torch.unsqueeze(x, 1)
print(x_reshaped)
print(x_reshaped.ndim)
print(x_reshaped.shape)
"""



"""
#View
b = x.view(4, 1)
print(x)
print(b)
# X shares the same memory address as the b view, so any changes to b will directly impact on X
b[3, 0] = 10
# or --> b[3][0] = 10
print(x)
print(b)
"""



"""
#Permuting
#Permutation is basically rearrangement
c = torch.randn(1, 3, 5)
c_permuted = torch.permute(c, (2, 0, 1)) # it will rearrange the shape of the tensor
print(c)
print(c.shape, "\n\n")
print(c_permuted)
print(c_permuted.shape)
"""