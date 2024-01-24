import torch
from torch import nn

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, inputs, targets):
        logits = self.token_embedding_table(inputs)
        return logits


