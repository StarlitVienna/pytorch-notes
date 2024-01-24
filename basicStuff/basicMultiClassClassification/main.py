from network import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def setup(prefer_cuda: bool) -> dict:
    setup_dict = {}
    if prefer_cuda:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    setup_dict["device"] = device
    return setup_dict

def numpy_to_torch(*Args, device):
    final_list = []
    for array in Args:
        final_list.append(torch.from_numpy(array).type(torch.float32).to(device))
    return final_list

def prepare_data(X, y, test_percent: float, seed: int, device):
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(X, y, test_size=test_percent, random_state=seed)
    return numpy_to_torch(train_inputs, test_inputs, train_labels, test_labels, device=device)

def gen_data(n_samples: int, classes: int, random_state: int, device) -> dict:
    data_dict = {}
    X, y = make_blobs(n_samples=n_samples, n_features=2, centers=classes, random_state=random_state)
    data_dict["train_inputs"], data_dict["test_inputs"], data_dict["train_labels"], data_dict["test_labels"] = prepare_data(
        X=X,
        y=y,
        test_percent=0.2,
        seed=42,
        device=device)
    return data_dict


def plot_data(train_inputs, test_inputs, train_labels, test_labels, preds: None) -> None:

    fig, axis = plt.subplots(2, 2)

    axis[0, 0].scatter(train_inputs[:, 0].cpu(), train_inputs[:, 1].cpu(), c=train_labels.cpu(), cmap=plt.cm.RdYlBu, s=16, label="Train Data")
    axis[0, 1].scatter(test_inputs[:, 0].cpu(), test_inputs[:, 1].cpu(), c=test_labels.cpu(), cmap=plt.cm.RdYlBu, s=16, label="Test Data")

    if preds is not None:
        axis[1, 0].scatter(test_inputs[:, 0].cpu(), test_inputs[:, 1].cpu(), c=preds.cpu(), cmap=plt.cm.RdYlBu, s=16, label="Preds Data")
    
    plt.show()

def train_model(model, loss_fn, optim, train_inputs, train_labels, epochs) -> None:
    model.train()
    for epoch in tqdm(range(epochs)):
        logits = model(train_inputs).squeeze()
        loss = loss_fn(logits, train_labels.type(torch.long))
        optim.zero_grad()
        loss.backward()
        optim.step()
    model.eval()

def main() -> None:
    setup_dict = setup(prefer_cuda = True)
    data_dict = gen_data(n_samples=500, classes=4, random_state=42, device=setup_dict["device"])

    model = MultiClassModel(device=setup_dict["device"], n_features=2, n_classes=4)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(params=model.parameters(), lr=0.1)

    train_model(
        model = model,
        loss_fn = loss_fn,
        optim = optim,
        train_inputs = data_dict["train_inputs"],
        train_labels = data_dict["train_labels"],
        epochs = 500
    )

    with torch.inference_mode():
        logits = model(data_dict["test_inputs"])
        probabilities = torch.softmax(logits, dim = 1)
        preds = torch.argmax(probabilities,dim = 1)
        print(preds)

    plot_data(
        train_inputs = data_dict["train_inputs"],
        test_inputs = data_dict["test_inputs"],
        train_labels = data_dict["train_labels"],
        test_labels = data_dict["test_labels"],
        preds=preds
    )
    

if __name__ == "__main__":
    main()

