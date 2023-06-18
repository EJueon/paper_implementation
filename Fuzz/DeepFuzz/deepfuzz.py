import sys
import os
import torch
import wandb
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
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

def train(train_dataset_path : str, dev_dataset_path: str, save_dir: str, max_seq_len : int = 50, vocab_size=98, batch_size=4096, lr=2e-4, epochs=50):
    check_dir(save_dir)
    train_dataset = pd.read_json(train_dataset_path)
    dev_dataset = pd.read_json(dev_dataset_path)
    wandb.init(project="deepfuzz")
    
    
    train_dataloader = DataLoader(DeepFuzzDataset(dataset=train_dataset, max_seq_len=max_seq_len, vocab_size=vocab_size), batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(DeepFuzzDataset(dataset=dev_dataset, max_seq_len=max_seq_len, vocab_size=vocab_size), batch_size=batch_size, shuffle=False)

    # 96: start, 97: end
    model = Seq2Seq(vocab_size=vocab_size, hidden_dim=512, n_layers=2, dropout=0.0)
    model.to('cuda')
    wandb.watch(model, model.loss_func, log="all", log_freq=10)
    optimizer = AdamW(params=model.parameters(), lr=lr, weight_decay=0.01, eps=1e-08,)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=0.001)
    
    for epoch in range(epochs):
        
        
        model.train()
        
        iter_bar = tqdm(train_dataloader)
        for batch in iter_bar:
            optimizer.zero_grad()
            batch = torch.stack([b.to("cuda") for b in batch])
            output, loss = model(batch)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            wandb.log({"epoch": epoch, "train loss": loss, "lr": optimizer.param_groups[0]["lr"] })
            iter_bar.set_description('Train Iter (lr=%5.5f, loss=%5.3f)'%(optimizer.param_groups[0]["lr"], loss.item()))
            torch.cuda.empty_cache()
            
        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch}"))
        model.eval()
        with torch.no_grad():
            iter_bar = tqdm(dev_dataloader)
            for batch in dev_dataloader:
                batch = torch.stack([b.to("cuda") for b in batch])
                output, loss = model(batch)
                wandb.log({"epoch": epoch, "eval loss": loss})
                iter_bar.set_description('Dev Iter (lr=%5.5f, loss=%5.3f)'%(optimizer.param_groups[0]["lr"], loss.item()))
                torch.cuda.empty_cache()
        
    
    
    
    
    
    
    return
    
def fuzz():
    return

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mode", required=True, type=str, choices=["preprocess", "train", "fuzz"])
    argparser.add_argument("--save_dir", required=False, type=str, default="./save")
    argparser.add_argument("--train_dataset_path", required=False, type=str, default="./save/train_dataset.json")
    argparser.add_argument("--dev_dataset_path", required=False, type=str, default="./save/dev_dataset.json")
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