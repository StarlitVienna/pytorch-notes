from network import *

def get_batch(block_size, batch_size, data):
    rand_text_place = torch.randint(0, len(data)-block_size, (batch_size,))
    batch_inputs = torch.stack([data[place:place+block_size] for place in rand_text_place])
    batch_labels = torch.stack([data[place+1:place+block_size+1] for place in rand_text_place])



def main():
    with open ("wizard", 'r', encoding="UTF-8") as f:
        text = f.read()
        chars_used = sorted(set(text))
    
    char_to_int = {char:integer for integer, char in enumerate(chars_used)}
    int_to_char = {integer:char for integer, char in enumerate(chars_used)}

    encode = lambda enc: [char_to_int[c] for c in enc]
    decode = lambda dec:"".join([int_to_char[i] for i in dec])

    data = torch.tensor(encode(text), dtype=torch.long)
    train_split = int(len(text)*0.8)
    train = data[:train_split]
    test = data[train_split:]

    get_batch(block_size=8, batch_size=4, data=train)




if __name__ == "__main__":
    main()