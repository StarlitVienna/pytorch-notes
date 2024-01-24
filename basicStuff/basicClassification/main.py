from network import *
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def gen_data():
    X, y = make_circles(n_samples=800, noise=0.03, random_state=42)
    return (X, y)

def plot_data(train_inputs, train_labels, test_inputs, test_labels, predictions):
    figure, axis = plt.subplots(2, 2)

    axis[0, 0].scatter(train_inputs[:, 0].cpu(), train_inputs[:, 1].cpu(), c=train_labels.cpu(), cmap=plt.cm.RdYlBu, s=16, label="Train Data")
    axis[0, 1].scatter(test_inputs[:, 0].cpu(), test_inputs[:, 1].cpu(), c=test_labels.cpu(), cmap=plt.cm.RdYlBu, s=16, label="Test Data")
    axis[1, 0].scatter(test_inputs[:, 0].cpu(), test_inputs[:, 1].cpu(), c=predictions.cpu(), cmap=plt.cm.RdYlBu, s=16, label="Preds Data")

    plt.legend(prop={"size": 14})
    plt.show()

def data_to_tensor(*data, device):
    final_list = []
    for d in data:
        final_list.append(torch.from_numpy(d).type(torch.float64).to(device))
    return final_list


def train_model(model, loss_fn, optim, train_inputs, train_labels, epochs):
    model.train()
    for epoch in range(epochs):
        logits = model(train_inputs)
        loss = loss_fn(logits, train_labels.unsqueeze(dim=1))
        optim.zero_grad()
        loss.backward()
        optim.step()
    model.eval()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    model = BasicClassificationModel()

    inputs, labels = gen_data()
    #Could split the data and turn it into tensor like that
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.25, random_state=42)
    train_inputs, test_inputs, train_labels, test_labels = data_to_tensor(train_inputs, test_inputs, train_labels, test_labels, device=device)


    optim = torch.optim.SGD(params=model.parameters(), lr=0.1)
    loss_fn = nn.BCEWithLogitsLoss()

    train_model(model,loss_fn, optim, train_inputs, train_labels, 500)

    with torch.inference_mode():
        logits = model(test_inputs)
        preds = torch.round(torch.sigmoid(model(test_inputs)))
        print(preds)

    plot_data(train_inputs, train_labels, test_inputs, test_labels, preds)






if __name__ == "__main__":
    main()