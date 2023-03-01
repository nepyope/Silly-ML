import torch
import torch.nn as nn
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#prepare input
worst_sentence = input("Type your sentence: ")
worst_sentence = ''.join(ch for ch in worst_sentence if ch.isalnum() or ch == ' ')
words = [x for x in worst_sentence.split(" ") if x]
words = ["<start>"] + words + ["<end>"]
worst_sentence = " ".join(words)

vocab = set(words)
vocab = {word: i for i, word in enumerate(set(words))}

train_iters = 1000
max_tokens = 100

class config:
    hidden_size=768
    vocab_size=len(vocab)
    key_size= 256
    lr = 0.001

def tokenizer(words, vocab):
    batch = torch.zeros(len(words), len(vocab))
    for i, word in enumerate(words):
        batch[i, vocab[word]] = 1
    return batch

def detokenizer(batch, vocab):
    sentence = ""
    for i in range(batch.shape[0]):
        word = list(vocab.keys())[list(vocab.values()).index(batch[i].argmax().item())]#access key from value
        sentence += word + " "
    return sentence[:-1]

class WorstTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.key_size = config.key_size

        self.Encoder = nn.Linear(self.vocab_size, self.hidden_size)
        self.Decoder = nn.Linear(self.hidden_size, self.vocab_size)

        self.Query = nn.Linear(self.hidden_size, self.key_size, bias=False)
        self.Key = nn.Linear(self.hidden_size, self.key_size, bias = False)
        self.Value = nn.Linear(self.hidden_size, self.hidden_size, bias = False)

        self.loss = 0
        self.lossf = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)

    def encode(self, x):
        x = self.Encoder(x)
        return x
    
    def forward(self, x): 
        x = self.Encoder(x)

        Q = self.Query(x)
        K = self.Key(x) 
        V = self.Value(x)
        A = Q@K.T
        y = A@V

        return self.Decoder(y)
    
    def decode(self, x):
        x = self.Decoder(x)
        return x
    
    def L(self):
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return self.loss
    
gpt = WorstTransformer(config).to(device)

#train
inp = words[:-1]
tar = words[1:]

for _ in tqdm(range(train_iters)):
    for t in range(len(inp)):#this should be a batch
        inp_t = tokenizer(inp[:t+1], vocab).requires_grad_(True).to(device)
        tar_t = tokenizer(tar[:t+1], vocab).to(device)
        pred_tar_t = gpt(inp_t)
        gpt.loss = gpt.lossf(pred_tar_t, tar_t)
        gpt.L()

#autoregressive inference
inp = ["<start>"]
while inp[-1] != "<end>" and len(inp) < max_tokens:
    inp_t = tokenizer(inp, vocab).requires_grad_(True).to(device)
    pred_t = detokenizer(gpt(inp_t),vocab).split(" ")
    inp.append(pred_t[-1])

print(" ".join(inp))