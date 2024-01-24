from network import *

#This is a bigram model
#It will try to predict the next letter based on the previous one (or predict a block of text based on the previous block of text)

def setup():
    setup_dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    setup_dict["device"] = device
    return setup_dict


def get_batch(block_size, batch_size, data, type, device):
    #train_random_indices = torch.randint(0, len(train_data)-block_size, (batch_size,))
    #test_random_indices = torch.randint(0, len(test_data)-block_size, (block_size,))
    random_indices = torch.randint(0, len(data)-block_size, (batch_size,))

    if type == "train":
        inputs = torch.stack([data[ri:ri+block_size] for ri in random_indices]).to(device)
        labels = torch.stack([data[ri+1:ri+block_size+1] for ri in random_indices]).to(device)
    else:
        inputs = torch.stack([data[ri:ri+block_size] for ri in random_indices]).to(device)
        labels = torch.stack([data[ri+1:ri+block_size+1] for ri in random_indices]).to(device)

    return (inputs, labels)



def gen_data(device):
    with open("./wizard", 'r', encoding="utf-8") as f:
        text = f.read()
        chars_used = sorted(set(text))
    vocab_size = len(chars_used)
    final_dict = {}
    index = 0


    # Character level tokenizer
    string_to_int = {char:integer for integer, char in enumerate(chars_used)}
    int_to_string = {integer:char for integer, char in enumerate(chars_used)}

    encode = lambda enc: [string_to_int[c] for c in enc]
    decode = lambda dec: "".join([int_to_string[i] for i in dec])

    data = torch.tensor(encode(text), dtype=torch.long)

    train_split = int(len(data)*0.8)
    train = data[:train_split]
    test = data[train_split:]


    block_size = 8
    batch_size = 4

    X_train, y_train = get_batch(block_size=block_size, batch_size=batch_size, data=train, type=train, device=device)
    print(X_train)
    print(y_train)

    """
    for t in range(block_size):
        context = train[:t+1]
        label = train[t+1]
        print("When context is ", context, "label is ", label)
    """

def main():
    setup_dict = setup()
    gen_data()

if __name__ == "__main__":
    main()