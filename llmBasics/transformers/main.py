from network import *
from tqdm import tqdm

#device = "cpu" if torch.cuda.is_available() else "cpu"
#device = "cuda" if torch.cuda.is_available() else "cpu"
#torch.set_default_device(device)
#print(f"Default device set to {device}")


class Controller():

    def train_test_split(self, data, train_percent):
        train_split = int(len(data)*train_percent)
        
        train_data = data[:train_split]
        test_data = data[train_split:]

        return (train_data, test_data)

    def tokenize(self):
        with open("input.txt", 'r', encoding="UTF-8") as f:
            text = f.read()
            self.chars = sorted(set(text))
            self.vocab_size = len(self.chars)
        
        self.stoi = {char:integer for integer, char in enumerate(self.chars)}
        self.itos = {integer:char for integer, char in enumerate(self.chars)}

        self.encode = lambda enc: [self.stoi[c] for c in enc]
        self.decode = lambda dec: "".join([self.itos[i] for i in dec])

        self.data = torch.tensor(self.encode(text))
        self.train_data, self.test_data = self.train_test_split(self.data, 0.9)

    def get_batch(self, data, batch_size, context_size):
        batch_ix = torch.randint(0, len(data)-context_size, (batch_size,))
        inputs = torch.stack([data[ix:ix+context_size] for ix in batch_ix], dim=0) #For a input tensor in the batch, there is an expected output tensor in the label, thus the need of +1 when generating the labels
        labels = torch.stack([data[ix+1:ix+1+context_size] for ix in batch_ix], dim=0)

        return (inputs, labels)

    def train_model(self, data, model, epochs, loss_fn, optimizer):
        for epoch in tqdm(range(epochs)):
            inputs, labels = self.get_batch(data, 64, 8)
            logits, loss = model(inputs, labels)
            #B, T, C = logits.shape
            #logits = logits.view(B*T, C)
            #labels = labels.view(B*T)
            #loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Final loss --> {loss}")

    def main(self):
        self.tokenize()
        xb, yb = self.get_batch(self.train_data, 2, 8)

        model = BigramModel(self.vocab_size)

        starting_idx_for_gen = torch.zeros((1, 1), dtype=torch.long)
        output = model.generate(starting_idx_for_gen, 10)
        print(self.decode(output[0].tolist()))

        optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        epochs = 10000
        model.train()
        self.train_model(self.train_data, model, epochs, loss_fn, optimizer)
        model.eval()
        with torch.inference_mode():
            output = model.generate(starting_idx_for_gen, 1000)
        print(self.decode(output[0].tolist()))



controller1 = Controller()
controller1.main()