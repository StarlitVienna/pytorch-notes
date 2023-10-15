import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


"""
#Scalar in tensor type
scalar = torch.tensor(9)
print(scalar.ndim) # print "dimensions" (more like the number of brackets rather than dimensions)

# get scalar back to python int
intScalar = scalar.item()
print(intScalar)
"""



"""
#Vector in tensor type 

vector = torch.tensor([6, 9])
print(vector.ndim) # number of dimensions
print(vector.shape) # overral shape of the vector
"""



"""
#Matrix in tensor type
matrix = torch.tensor([[1, 2], [3, 4]])
print(matrix)
print(matrix.ndim) # number of "dimensions" of the matrix (2)
print(matrix.shape) # overrall shape of the matrix (2x2)
"""


"""
#Tensor in tensor type
tensor = torch.tensor(
    [[[1,2,3],
      [4,5,6]],
    [ [7,8,9],
      [10,11,12]]])

print(tensor)
print(tensor.ndim)
print(tensor.shape)
"""