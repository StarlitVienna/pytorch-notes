from model import *
from tqdm import trange

def train_model(model, optimizer, loss_fn, inputs, labels, epochs):
    model.train()
    for epoch in trange(epochs):
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()

def main_func():

    model = CircleDataModel()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    train_model(model, optimizer, loss_fn, X_train, Y_train, 10000)

    with torch.inference_mode():
        preds = torch.softmax(model(X_test), dim=1).argmax(dim=1)
        acc = torch.sum(torch.eq(preds, Y_test))/len(Y_test)
        print(acc)




if __name__ == "__main__":
    main_func()