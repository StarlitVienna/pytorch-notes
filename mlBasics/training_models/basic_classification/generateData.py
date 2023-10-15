import torch
import sklearn
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

N_SAMPLES = 1000
X, y = make_circles(N_SAMPLES, noise=0.03, random_state=42)

X_train, X_test, Y_train, Y_test = train_test_split(X,y, train_size=0.8, random_state=42, shuffle=True)

X_train = torch.from_numpy(X_train).to(device).double()
X_test = torch.from_numpy(X_test).to(device).double()
Y_train = torch.from_numpy(Y_train).to(device).double()
Y_test = torch.from_numpy(Y_test).to(device).double()

#print(X_train.dtype) # float64

#print(len(X[:, 0]))
#print(torch.from_numpy(y).unsqueeze(dim=1))

#X_train, X_test, Y_train, Y_test = 

"""
def plot_stuff(X_train, Y_train, X_test, Y_Test, preds):
    plt.figure(figsize=(10, 7))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=4, cmap=plt.cm.RdYlBu, label="Train data")
    #plt.scatter(X_test[:, 0], X_test[:, 1], c='g', s=4, cmap=plt.cm.RdYlBu, label="Test data")
    #plt.scatter(X_train, preds, c='r', s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()

plot_stuff(X_train, Y_train, X_test, Y_test, None)
"""

if __name__ == "__main__":
    figure, axis = plt.subplots(2,2)
    axis[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=4, cmap=plt.cm.RdYlBu, label="Train data")
    axis[0, 1].scatter(X_test[:, 0], X_test[:, 1], c=Y_test, s=4, cmap=plt.cm.RdYlBu, label="Train data")
    plt.legend(prop={"size": 14})
    plt.show()