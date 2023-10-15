import torch

torch.manual_seed(69)


rand_matrix1 = torch.randint(1, 5, [3, 3])
rand_matrix2 = torch.randint(1, 5, [3, 3])

print(rand_matrix1, "\n")
print(rand_matrix2, "\n")

"""
#This wont do matrix multiplication, it will just multiply each respective item
mult_result = torch.mul(rand_matrix1, rand_matrix2)
print(mult_result)
"""

mult_result = torch.mm(rand_matrix1, rand_matrix2) # .mm or .matmul

print(mult_result)

#tensor.T to get a view of the transposed tensor