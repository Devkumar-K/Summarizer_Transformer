

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class SummarizationDataset(Dataset):
    

    def __init__(self, hf_dataset, tokenizer, max_src_len=512, max_tgt_len=128):
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        
        self.pad_id = tokenizer.pad_token_id     
        self.bos_id = tokenizer.cls_token_id     
        self.eos_id = tokenizer.sep_token_id     

        
        articles   = list(hf_dataset["article"])
        highlights = list(hf_dataset["highlights"])

        
        print(f"    Tokenising {len(articles):,} articles …")
        src_enc = tokenizer(
            articles,
            truncation=True,
            max_length=max_src_len,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        self.all_src = src_enc["input_ids"]       

        print(f"    Tokenising {len(highlights):,} summaries …")
        tgt_enc = tokenizer(
            highlights,
            truncation=True,
            max_length=max_tgt_len - 1,           
            add_special_tokens=False,
            return_attention_mask=False,
        )
        self.all_tgt = tgt_enc["input_ids"]

        print(f"    ✓ {len(self)} samples ready.")

    
    def __len__(self):
        return len(self.all_src)

    def __getitem__(self, idx):
        src_tokens = self.all_src[idx]            
        tgt_tokens = self.all_tgt[idx]

       
        src_ids = src_tokens + [self.pad_id] * (self.max_src_len - len(src_tokens))

    
        dec_in = [self.bos_id] + tgt_tokens
        dec_in = dec_in + [self.pad_id] * (self.max_tgt_len - len(dec_in))

        
        dec_tgt = tgt_tokens + [self.eos_id]
        dec_tgt = dec_tgt + [self.pad_id] * (self.max_tgt_len - len(dec_tgt))

        return {
            "src_ids":    torch.tensor(src_ids,  dtype=torch.long),
            "dec_input":  torch.tensor(dec_in,   dtype=torch.long),
            "dec_target": torch.tensor(dec_tgt,  dtype=torch.long),
        }



def build_loaders(
    tokenizer,
    max_src_len: int = 512,
    max_tgt_len: int = 128,
    train_samples: int = 40_000,
    batch_size: int = 16,
    num_workers: int = 0,       
    seed: int = 42,
):
    """
    Returns
        train_loader, val_loader
    """
    print("Loading CNN/DailyMail 3.0.0 …")
    raw = load_dataset("cnn_dailymail", "3.0.0")

    
    train_raw = raw["train"].shuffle(seed=seed).select(range(train_samples))
    val_raw   = raw["validation"]
    print(f"  Train samples : {len(train_raw):,}")
    print(f"  Val   samples : {len(val_raw):,}")

    
    print("  Building training set …")
    train_ds = SummarizationDataset(train_raw, tokenizer, max_src_len, max_tgt_len)
    print("  Building validation set …")
    val_ds   = SummarizationDataset(val_raw,   tokenizer, max_src_len, max_tgt_len)

    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,          
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader