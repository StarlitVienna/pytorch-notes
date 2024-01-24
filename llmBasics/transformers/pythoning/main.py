from model import *

class controller():
    def train_test_split(self, data, train_percent, test_percent):
        train_split = int(len(data)*train_percent)
        test_split = int(len(data)*test_percent)

        train = data[:train_split]
        test = data[train_split:train_split+test_split]

        print(train.shape)
        print(test.shape)

        return (train, test)


    def tokenize(self):
        with open("input.txt", 'r', encoding="UTF-8") as f:
            self.text = f.read()
            words = self.text.splitlines()
            #chars = sorted(set("".join(words))) #Need to also store the new line unicode, thus this line wont work
            chars = sorted(set(self.text))
        
        self.stoi = {char:integer for integer, char in enumerate(chars)}
        self.itos = {integer:char for integer, char in enumerate(chars)}

        self.vocab_size = len(self.stoi)

        self.encode = lambda enc: [self.stoi[c] for c in enc]
        self.decode = lambda dec: "".join([self.itos[i] for i in dec])

        encoded = self.encode(list("hii there"))
        decoded = self.decode(encoded)

        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)

        self.train, self.test = self.train_test_split(self.data, 0.9, 0.1)

    def get_batch(self, data, batch_size, context_size):
        batch_ix = torch.randint(0, len(data)-context_size, (batch_size, ))
        x = torch.stack([data[i:i+context_size] for i in batch_ix])
        y = torch.stack([data[i+1:i+context_size+1] for i in batch_ix])
        print(x[:5])
        print(y[:5])

        for b in range(batch_size):
            for t in range(context_size):
                print(f"for this input --> {x[b, :t+1]}")
                print(f"expected output is --> {y[b, t]}")

        return (x, y)


    def main(self):
        self.tokenize()
        Xtrain, Ytrain = self.get_batch(data=self.train, batch_size=4, context_size=8)


        model = BigramLanguageModel(self.vocab_size)
        out = model(Xtrain, Ytrain)
        generation = model.generate(torch.zeros((1, 1), dtype=torch.long), 10)


if __name__ == "__main__":
    control = controller()
    control.main()