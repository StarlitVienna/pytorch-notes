import matplotlib.pyplot as plt
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

#Basic linear regression dataset

weight = 0.7
bias = 0.3

start = 0
end = 1
step_size = 0.02
x = torch.arange(start=start, end=end, step=step_size).unsqueeze(dim=1)
y = torch.mul(x, weight) + bias

#print(x)
#print(y)

#Spliting the data
training_percentage = 0.8

train_split = int(len(x)*training_percentage)
X_train = x[:train_split]
Y_train = y[:train_split]

X_test = x[train_split:]
Y_test = y[train_split:]


def plot_stuff(x_train, y_train, x_test, y_test, predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(x_train, y_train, c="b", s=4, label="Train data")
    plt.scatter(x_test, y_test, c="g", s=4, label="Test data")
    plt.scatter(x_test, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14})
    plt.show()



