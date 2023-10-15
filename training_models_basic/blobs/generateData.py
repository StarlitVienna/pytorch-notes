import torch
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

N_SAMPLES = 1000
CLASSES = 4
MANUAL_SEED = 42


X, Y = make_blobs(N_SAMPLES, n_features=2, centers=CLASSES, random_state=MANUAL_SEED)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=MANUAL_SEED, shuffle=True)


X_train = torch.from_numpy(X_train).to(device).type(torch.float32)
X_test = torch.from_numpy(X_test).to(device).type(torch.float32)
Y_train = torch.from_numpy(Y_train).to(device).type(torch.long) # cannot work with float labels for cross entropy loss
Y_test = torch.from_numpy(Y_test).to(device).type(torch.long)

print(Y_train)

"""
#It's best if both dtypes meet
print(X_train.dtype) 
print(Y_train.dtype)
"""


"""
if __name__ == "__main__":
    X_train = X_train.cpu()
    Y_train = Y_train.cpu()
    X_test = X_test.cpu()
    Y_test = Y_test.cpu()
    figure, axis = plt.subplots(2,2)
    axis[0,0].scatter(X_train[:,0], X_train[:, 1], c=Y_train, cmap=plt.cm.RdYlBu)
    axis[0,1].scatter(X_test[:,0], X_test[:, 1], c=Y_test, cmap=plt.cm.RdYlBu)
    plt.show()
"""