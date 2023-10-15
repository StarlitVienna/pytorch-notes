import matplotlib.pyplot as plt
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

W = 0.1234
B = 0.345

X = torch.arange(start=0, end=5, step=0.02).unsqueeze(dim=1)
Y = (X * W) + B

train_split = int(0.8 * len(X))

X_train = X[:train_split]
Y_train = Y[:train_split]

X_test = X[train_split:]
Y_test = Y[train_split:]

def plot_data(X_train, Y_train, X_test, Y_test, predicts):
    plt.figure(figsize=(10, 7))
    plt.scatter(X_train.cpu(), Y_train.cpu(), c='b', s=4, label="Train data")
    plt.scatter(X_test.cpu(), Y_test.cpu(), c='g', s=4, label="Test data")
    plt.scatter(X_test.cpu(), predicts.cpu(), c='r', s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()