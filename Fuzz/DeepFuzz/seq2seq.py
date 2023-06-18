import torch
import random
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

class DeepFuzzDataset(Dataset):
    def __init__(self, dataset, max_seq_len, vocab_size):
        super().__init__()
        self.dataset = dataset["data"]
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len + 1
    
    
    def char_to_ord(self, sentence):
        data = [96]
        for s in sentence:
            if ord(s) < self.vocab_size:
                data.append(ord(s)-32)
        data.append(97)
        return data
    
    def pad_id(self, x):
        if len(x) < self.max_seq_len:
            pad = [0] * (self.max_seq_len - len(x))
            x.extend(pad)
        return x
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.char_to_ord(self.dataset[idx])
    
        x = F.one_hot(torch.tensor(self.pad_id(data[:-1]), dtype=torch.long), num_classes=self.vocab_size).float()
        y = F.one_hot(torch.tensor(self.pad_id(data[1:]), dtype=torch.long), num_classes=self.vocab_size).float()
        return (x, y)
        

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_dim, n_layers, dropout=dropout)
    
    def forward(self, x):
        # x :  (batch_size, max_seq_len)
        embed = x.transpose(0, 1) # (max_seq_len, batch_size)
        logits, (h, c) = self.lstm(embed) # logits : (max_seq_len, batch_size, embed_dim), h, c: (n_layer, batch_size, embed_dim)
        return logits, h, c

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_dim, n_layers, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, h, c):
        # print(h.shape, c.shape, x.shape)
        logits, (h, c) = self.lstm(x.transpose(0, 1), (h, c))
        pred = self.classifier(logits) # (1, batch_size, vocab_size)
        return pred, h, c
             

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers=2, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = Encoder(vocab_size, hidden_dim, n_layers, dropout)
        self.decoder = Decoder(vocab_size, hidden_dim, n_layers, dropout)
        self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, data, teacher_forcing_ratio=1.0):
        x, y = data
        
        _, h, c = self.encoder(x)
        preds = torch.zeros_like(x)
        sub_x = None
        loss = 0
        sub_x = x[:, 0:1, :]
        for i in range(1, x.shape[1]):
            pred, h, c = self.decoder(sub_x, h, c)
            pred = pred.transpose(0,1) 
            preds[:, i, :] = pred[0]
            teacher_forcing = random.random() < teacher_forcing_ratio
            
            pred = pred.argmax(2)
            pred = F.one_hot(pred, self.vocab_size).float()

            sub_x = x[:, i:i+1, :] if teacher_forcing else pred
            loss += self.loss_func(preds[:, i, :].to('cuda'), y[:, i])
        loss = loss / x.shape[1]
        return preds, loss