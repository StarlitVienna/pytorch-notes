import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor # image to tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
#torch.set_default_device(device)


BATCH_SIZE = 32


train_data = datasets.FashionMNIST(root="./datasets/train-fashionMNIST", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="./datasets/test_fashionMNIST", train=False, download=True, transform=ToTensor())



class_names = train_data.classes
#print(len(test_data))
image, label = train_data[0]
#print(image.shape)
#print(class_names[label])

#train_data = train_data.data.to("cuda")
#test_data = test_data.data.to("cuda")

#train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device="cuda"))
#test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device="cuda"))

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)


#print(next(iter(train_dataloader)))
#print(next(iter(train_dataloader.dataset)))


#print(class_names[7])
#plt.imshow(image.squeeze(), cmap="gray")
#plt.show()
"""
if __name__ == "__main__":
    figure, axis = plt.subplots(2,2)
    plt.show()
"""