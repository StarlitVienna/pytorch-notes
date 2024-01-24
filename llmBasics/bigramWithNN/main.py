import torch
from torch import nn
import torch.nn.functional as F

with open("names.txt", "r", encoding="UTF-8") as f:
    text = f.read()
    words = text.splitlines()
    chars = sorted(set("".join(words)))

stoi = {char:integer+1 for integer, char in enumerate(chars)}
itos = {integer+1:char for integer, char in enumerate(chars)}

