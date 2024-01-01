import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

N_SAMPLES = 100
NOISE = 0.02
SEED = 42



X, y = make_circles(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=SEED)


X_train = torch.from_numpy(X_train).type(torch.float64).to(device)
X_test = torch.from_numpy(X_test).type(torch.float64).to(device)
Y_train = torch.from_numpy(Y_train).to(device)
Y_test = torch.from_numpy(Y_test).to(device)

if __name__ == "__main__":
    figure, axis = plt.subplots(2,2)
    axis[0,0].scatter(X_train[:, 0].cpu(), X_train[:, 1].cpu(), c=Y_train.cpu(), cmap=plt.cm.RdYlBu)
    axis[0,1].scatter(X_test[:, 0].cpu(), X_test[:, 1].cpu(), c=Y_test.cpu(), cmap=plt.cm.RdYlBu)
    plt.show()