from model import *
from tqdm import trange


def train_model(model, loss_fn, optim, inputs, labels, epochs):
    model.train()
    for epoch in trange(epochs):
        preds = model(inputs)
        loss = loss_fn(preds, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
    model.eval()


def main_func():
    torch.manual_seed(42)
    model = BasicLinearRegressionModel()
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

    train_model(model, loss_fn, optimizer, X_train, Y_train, 10000)

    with torch.inference_mode():
        preds = model(X_test)

    print("Final params:", model.state_dict())
    print("\n Desired weights --> ", W)
    print("Desired bias --> ", B)
    plot_data(X_train, Y_train, X_test, Y_test, preds)


if __name__ == "__main__":
    main_func()