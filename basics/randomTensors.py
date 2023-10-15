import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print("Current device in use --> ", device)
print("\n\n")

"""
#Create random tensor of size (x, y)
x = 3
y = 5
random_tensor = torch.randn(size=(x,y))
print(random_tensor)
print(random_tensor.ndim)
print(random_tensor.shape)
"""

"""
#Tensor of rgb
random_image_tensor = torch.randn(size=(3, 20, 20))
#plt.imshow(random_image_tensor.cpu().numpy()[0], cmap="YlGnBu")
#plt.show()
"""