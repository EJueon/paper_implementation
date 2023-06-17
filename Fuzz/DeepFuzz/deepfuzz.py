import sys
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
import pandas as pd

from seq2seq import Seq2Seq, DeepFuzzDataset

def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

def preprocess(dataset_path : str, save_dir: str, max_seq_len : int = 50) -> pd.DataFrame:
    """
    In DeepFuzz, preprocessing stages such as handling comments, whitespace, and Macros are performed. 
    However, in this implementation, we proceed under the assumption that the data has already been preprocessed. 
    The only preprocessing we do is to split the data into sections of a maximum sequence length.
    """
    
    check_dir(save_dir)
    dataset = pd.read_json(dataset_path)

    sequences = []
    for data in tqdm(dataset["data"]):
        pointer = 0
        for i in range(0, len(data), max_seq_len):
            pointer = i
            sequences.append(data[i:i+max_seq_len])
        if pointer < len(data):
            sequences.append(data[pointer:len(data)])
    
    new_dataset = pd.DataFrame(sequences, columns=["data"])
    new_dataset.to_json(os.path.join(save_dir, os.path.basename(dataset_path)), orient="records", force_ascii=False)

def train(train_dataset_path : str, dev_dataset_path: str, save_dir: str, max_seq_len : int = 50, batch_size=64, lr=0.001, epochs=100):
    check_dir(save_dir)
    train_dataset = pd.read_json(train_dataset_path)
    dev_dataset = pd.read_json(dev_dataset_path)
    
    train_dataloader = DataLoader(DeepFuzzDataset(dataset=train_dataset), batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(DeepFuzzDataset(dataset=dev_dataset), batch_size=batch_size, shuffle=False)
    
    model = Seq2Seq(98, 512, 512, 2, 0.2)
    optimizer = AdamW(lr=lr, weight_decay=0.01, eps=1e-08)
    for epoch in tqdm(range(epochs)):
        
        optimizer.zero_grad()
        model.train()
        for batch in train_dataloader:
            output, loss = model(batch)
            loss.backward()
            optimizer.step()
            print(loss)
            
        model.eval()
        for batch in dev_dataloader:
            output, loss = model(batch)
            print(loss)
            
        
    
    
    
    
    
    
    return
    
def fuzz():
    return

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mode", required=True, type=str, choices=["preprocess", "train", "fuzz"])
    argparser.add_argument("--save_dir", required=False, type=str, default="./save")
    argparser.add_argument("--train_dataset_path", required=False, type=str, default="")
    argparser.add_argument("--dev_dataset_path", required=False, type=str, default="")
    argparser.add_argument("--max_seq_len", required=False, type=int, default=50)
    args = argparser.parse_args(sys.argv[1:])
    
    if args.mode == "preprocess":
        if args.train_dataset_path:
            preprocess(args.train_dataset_path, args.save_dir, max_seq_len=args.max_seq_len)
        if args.dev_dataset_path:
            preprocess(args.dev_dataset_path, args.save_dir, max_seq_len=args.max_seq_len)
    elif args.mode == "train":
        train(args.train_dataset_path, args.dev_dataset_path, args.save_dir)
    elif args.mode == "fuzz":
        fuzz()