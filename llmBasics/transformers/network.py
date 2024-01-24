import torch
from torch import nn
import torch.nn.functional as F

#device = "cpu" if torch.cuda.is_available() else "cpu"
#device = "cuda" if torch.cuda.is_available() else "cpu"
#torch.set_default_device(device)
#print(f"Default device set to {device}")

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, labels=None):
        #print(self.token_embedding_table)
        logits = self.token_embedding_table(idx) #It takes in batches
        #Logits are B x T x C (batches by inputs by channels)
        #pytorch cross entropy loss function expects C as the second dimension


        if labels is not None:
            B, T, C = logits.shape

            #Self attention
            #xbow --> x bag of words
            xbow = torch.zeros((B, T, C))

            logits = logits.view(B*T, C)
            labels = labels.view(B*T)

            loss = F.cross_entropy(logits, labels)

            return (logits, loss)
        else:
            return logits
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:, -1, :] #get the logits of the last batch only
            #print(logits[:, -1, :] == logits[0, :, :])
            probs = torch.softmax(logits, dim=1)
            sample = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, sample), dim=1)
        return idx #Return prediction