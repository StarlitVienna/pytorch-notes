import matplotlib.pyplot as plt
from model import *
from tqdm import trange

def train_model(loss_fn, optimizer, train_data, train_labels, model, epochs):
    model.train()
    for epoch in trange(epochs, desc="training model"):
        preds = model(train_data)
        loss = loss_fn(preds, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()


def main_func():
    torch.manual_seed(42)
    model = LinearRegressionModel()

    loss_fn = nn.L1Loss() #Mean absolute error
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)
    train_model(loss_fn, optimizer, X_train, Y_train, model, 10000)
    print(f"Ending parameters: {model.state_dict()}")

    with torch.inference_mode():
        predicts = model(X_test)
        #With inference mode there is no need to do predicts.detach() beacuse it will already detach it from the gradient

    plot_stuff(X_train.cpu(), Y_train.cpu(), X_test.cpu(), Y_test.cpu(), predicts.cpu())
    #torch.save(model.state_dict(), "path/filename.pth") to save the 
    #Load the model
    #model.load_state_dict(torch.load(f="path/filename.pth"))



if __name__ == "__main__":
    main_func()