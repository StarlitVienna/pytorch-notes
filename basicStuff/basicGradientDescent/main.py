from neuralnetwork import *
import matplotlib.pyplot as plt
import pickle

def setup():
    setup_dict = {}

    version = torch.__version__
    device = "cuda" if torch.cuda.is_available() else "cpu"

    setup_dict["version"] = version
    setup_dict["device"] = device
    return setup_dict

def generate_data():
    weight = 0.7
    bias = 0.6
    start = 0
    end = 1
    step = 0.01

    #arange does not include the last specified number or the last step number if it's higher than the end value
    inputs = torch.arange(start=start, end=end, step=step).unsqueeze(dim=1)
    expected_outputs = inputs * weight + bias
    return (inputs, expected_outputs)

def split_data(features, labels, train_split_percent):
    train_amount = int(train_split_percent*len(features))

    train_data = features[:train_amount]
    test_data = features[train_amount:]

    train_labels = labels[:train_amount]
    test_labels = labels[train_amount:]

    return (train_data, test_data, train_labels, test_labels)

def plot_data(train_data, test_data, train_labels, test_labels, predictions = None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data.cpu(), train_labels.cpu(), c="r", s=4, label="train")
    plt.scatter(test_data.cpu(), test_labels.cpu(), c="g", s=4, label="test")
    
    if predictions is not None:
        plt.scatter(test_data.cpu(), predictions.cpu(), c="b", s=4, label="predictions")

    plt.legend(prop={"size": 14})
    plt.show()

def train_model(model, epochs, loss_fn, optimizer, train_data, train_labels):
    model.train()
    for epoch in range(epochs):
        preds = model(train_data)
        loss = loss_fn(preds, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    print(loss)
    pass

def save_model_params(model):
    torch.save(model.state_dict(), "basicLinearRegressionModelStateDict.pt")
    return 

def main():
    setup_dict = setup()
    torch.set_default_device(setup_dict["device"])
    X, y = generate_data()
    train_data, test_data, train_labels, test_labels = split_data(features=X, labels=y, train_split_percent=0.8)

    model = LinearRegressionModel()

    #The loss function that is going to be used for this is the L1Loss wich is the MAE (mean absolute error)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01) #Stochastic gradient descent
    #0.01 LR is just too low for 100 epochs, might need around 500-1000 epochs for it to work
    train_model(
        model=model, 
        epochs=1000, 
        loss_fn=loss_fn, 
        optimizer=optimizer,
        train_data=train_data,
        train_labels=train_labels)

    with torch.inference_mode():
        predictions = model(test_data)

    plot_data(train_data, test_data, train_labels, test_labels, predictions)
    save_model_params(model=model)

if __name__ == "__main__":
    main()