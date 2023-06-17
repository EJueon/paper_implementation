import torch
import random
from torch import nn
from torch.utils.data import Dataset

class DeepFuzzDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    
    def char_to_ord(self, sentence):
        data = []
        for s in sentence:
            data.append(ord(s))
        return data
    
    def __len__(self):
        return len(self.dataset)
    
    def get_item(self, idx):
        return self.char_to_ord(self.dataset[idx])
        

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
    
    def forward(self, x):
        embed = self.embedding(x)
        logits, (h, c) = self.lstm(embed)
        return logits, h, c

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, h, c):
        embed = self.embedding(x)
        logits, (h, c) = self.lstm(embed, (h, c))
        pred = self.classifier(logits) # check 필요
        return pred, h, c
             

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=2, dropout=0.2):
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim, n_layers, dropout)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim, n_layers, dropout)
        self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, x, teacher_forcing_ratio=0.5):
        
        _, h, c = self.encoder(x)
        
        preds = torch.zeros_like(x)
        
        sub_x = None
        for i in range(0, len(x)):
            sub_x = x[:, 0]
            pred, h, c = self.decoder(sub_x, h, c)
            
            preds[:, i, :] = pred
            teacher_forcing = random.random() < teacher_forcing_ratio
            
            sub_x = x[i] if teacher_forcing else preds.argmax(1)

        loss = self.loss_func(preds, x)
        return preds, loss