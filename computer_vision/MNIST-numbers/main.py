from model import *
from tqdm import tqdm
from tqdm import trange


def train_model(model, optimizer, loss_fn, dataloader, epochs):
    model.train()
    for epoch in range(epochs):
        for batch, (X, Y) in enumerate(tqdm(dataloader)):
            logits = model(X.to(device))
            loss = loss_fn(logits, Y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()

def predict_image(model, index):
    with torch.inference_mode():
        #print(train_data[0])
        #dummy_batch = torch.randn([32, 1, 28, 28])
        #logits = model(dummy_batch)
        #print(logits.shape)
        image, image_label = test_data[index]
        logits = model(image.to("cuda"))
        preds = torch.softmax(logits, dim=1).argmax(dim=1)
        print(class_names[preds])

    plt.imshow(image.squeeze())
    plt.show()

    predict_something = int(input("Select index to predict (0-9999): "))
    predict_image(model, predict_something)


def main_func():
    model = FashionMNISTModel().to("cuda")

    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    
    #train_data = train_data.data.to("cuda")
    #test_data = test_data.data.to("cuda")

    train_model(model, optimizer, loss_fn, train_dataloader, 3)

    predict_something = int(input("Select index to predict (0-9999): "))
    predict_image(model, predict_something)


if __name__ == "__main__":
    main_func()