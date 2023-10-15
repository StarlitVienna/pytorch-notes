from model import *
from tqdm import trange

def train_model(model, optimizer, loss_fn, inputs, labels, epochs):
    model.train()
    for epoch in trange(epochs):
        logits = model(inputs)
        loss = loss_fn(logits, labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()

def calc_acc(preds, labels):
    return torch.sum(preds==labels)/len(labels)*100

def main_func():
    model = BlobSolvingNN()

    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    train_model(model, optimizer, loss_fn, X_train, Y_train, 1000)

    with torch.inference_mode():
        logits = model(X_test)
        preds_probs = torch.softmax(logits, dim=1)
        #preds = torch.argmax(preds_probs, dim=1)
        preds = preds_probs.argmax(dim=1)
        #print(preds)
        print(f"Accuracy --> {calc_acc(preds, Y_test)}%")

    X_trainCPU = X_train.cpu()
    Y_trainCPU = Y_train.cpu()
    X_testCPU = X_test.cpu()
    Y_testCPU = Y_test.cpu()
    figure, axis = plt.subplots(2,2)
    axis[0,0].scatter(X_trainCPU[:,0], X_trainCPU[:, 1], c=Y_trainCPU, cmap=plt.cm.RdYlBu)
    axis[0,1].scatter(X_testCPU[:,0], X_testCPU[:, 1], c=Y_testCPU, cmap=plt.cm.RdYlBu)
    axis[1,0].scatter(X_testCPU[:,0], X_testCPU[:, 1], c=preds.cpu(), cmap=plt.cm.RdYlBu)
    plt.show()


    pass

if __name__ == "__main__":
    main_func()