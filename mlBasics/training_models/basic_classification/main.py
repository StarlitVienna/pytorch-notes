from model import *
from tqdm import trange

def train_model(model, optim, loss_fn, inputs, labels, epochs):
    model.train()
    for epoch in trange(epochs):
        logits = model(inputs).squeeze()
        #print(logits)
        loss = loss_fn(logits, labels) #BCEWithLogitsLoss will already apply Sigmoid to it
        optim.zero_grad()
        loss.backward()
        optim.step()
        pass
    model.eval()

def calc_acc(preds, labels):
    return torch.sum(preds==labels)/len(labels)

def main_func():

    model = BasicClassificationModel().to(device)
    #print(next(model.parameters()).device)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
    loss_fn = nn.BCEWithLogitsLoss()

    train_model(model, optimizer, loss_fn, X_train, Y_train.type(torch.float64), 1000)

    with torch.inference_mode():
        logits = model(X_test)
        preds = torch.round(torch.sigmoid(logits))
        print(preds.squeeze())
        print(Y_test)
        print(calc_acc(preds.squeeze(), Y_test))
        #print(logits) # logits are the inputs for the sigmoid
        #print(torch.sigmoid(logits))


        #print(preds.shape)
        #print(preds)
        #preds = torch.argmax(logits, dim=1) # for more than 1 out features
        #print(preds)
        #print(preds == Y_train)
        #acc = torch.sum(preds == Y_train)/len(Y_train)
        #print(acc)
    pass

if __name__ == "__main__":
    main_func()