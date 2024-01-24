from network import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

def setup():
    setup_dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    setup_dict["device"] = device
    return setup_dict

def numpy_to_tensor(*Args, device):
    final_list = []
    for array in Args:
        final_list.append(torch.from_numpy(array).to(device).type(torch.float64))
    return final_list


def gen_data(n_samples: int, noise: float, random_state:int, test_percent: float, device) -> list:
    X, y = make_circles(n_samples=n_samples, noise=noise, random_state=random_state)
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(X, y, test_size=test_percent, random_state=random_state)
    return numpy_to_tensor(train_inputs, test_inputs, train_labels, test_labels, device=device)

def plot_data(train_inputs, test_inputs, train_labels, test_labels, preds):
    fig, axis= plt.subplots(2, 2)

    axis[0, 0].scatter(train_inputs[:, 0].cpu(), train_inputs[:, 1].cpu(), c=train_labels.cpu(), cmap=plt.cm.RdYlBu, s=16, label="train_data")
    axis[0, 1].scatter(test_inputs[:, 0].cpu(), test_inputs[:, 1].cpu(), c=test_labels.cpu(), cmap=plt.cm.RdYlBu, s=16, label="test_data")
    axis[1, 0].scatter(test_inputs[:, 0].cpu(), test_inputs[:, 1].cpu(), c=preds.cpu(), cmap=plt.cm.RdYlBu, s=16, label="preds")

    plt.show()

def train_model(model, loss_fn, optim, train_inputs, train_labels, epochs):
    model.train()
    for epoch in tqdm(range(epochs)):
        logits = model(train_inputs)
        loss = loss_fn(logits, train_labels.unsqueeze(dim=1))
        optim.zero_grad()
        loss.backward()
        optim.step()
    model.eval()

def main():
    setup_dict = setup()
    train_inputs, test_inputs, train_labels, test_labels = gen_data(n_samples=500, noise=0.03, random_state=42, test_percent=0.2, device=setup_dict["device"])

    model = BasicBinaryClassificationModel(device=setup_dict["device"])
    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.SGD(params=model.parameters(), lr=0.1)
    train_model(model=model, loss_fn=loss_fn,optim=optim, train_inputs=train_inputs, train_labels=train_labels, epochs=500)

    with torch.inference_mode():
        logits = model(test_inputs)
        preds = torch.round(torch.sigmoid(logits))

    plot_data(train_inputs=train_inputs, test_inputs=test_inputs, train_labels=train_labels, test_labels=test_labels, preds=preds)


if __name__ == "__main__":
    main()