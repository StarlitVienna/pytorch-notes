import torch
from torch import nn
import torch.nn.functional as F
def setup():
    setup_dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    setup_dict["device"] = device
    print(f"current device set to --> {device}")

    return setup_dict

def tokenizer():
    with open("./names.txt", 'r', encoding="UTF-8") as f:
        text = f.read()
        words = text.splitlines()
        chars = sorted(set("".join(words)))

    stoi = {char:integer+1 for integer,char in enumerate(chars)}
    itos = {integer+1:char for integer,char in enumerate(chars)}

    stoi["."] = 0
    itos[0] = "."

    return (stoi, itos, words)

def gen_data(words):
    inputs, labels = [], []
    for w in words[:]:
        for char in w + ".":




def main():
    stoi, itos, words = tokenizer()
    vocab_size = len(stoi)
    C = torch.randn((vocab_size, 2))
    gen_data(words)





if __name__ == "__main__":
    main()