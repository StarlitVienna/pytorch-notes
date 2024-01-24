import torch
from torch import nn



class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()