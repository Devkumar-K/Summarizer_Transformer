import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from model   import Transformer, count_parameters
from dataset import build_loaders


class Config:
    
    d_model          = 256
    n_heads          = 4
    n_enc_layers     = 4
    n_dec_layers     = 4
    d_ff             = 1024
    dropout          = 0.1
    max_enc_len      = 512
    max_dec_len      = 128

    
    batch_size       = 16
    lr               = 3e-4
    epochs           = 10
    grad_clip        = 1.0

    
    train_samples    = 40_000
    seed             = 42

    
    tokenizer_name   = "bert-base-uncased"

    
    save_path        = "transformer_summ.pt"



def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



def train_one_epoch(model, loader, optimiser, criterion, device, epoch, clip):
    model.train()
    running_loss = 0.0
    n_batches = len(loader)

    for i, batch in enumerate(loader, 1):
        src        = batch["src_ids"].to(device)       
        dec_input  = batch["dec_input"].to(device)     
        dec_target = batch["dec_target"].to(device)    

        
        logits = model(src, dec_input)                

        
        logits     = logits.reshape(-1, logits.size(-1))  
        dec_target = dec_target.reshape(-1)                

        loss = criterion(logits, dec_target)

       
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimiser.step()

        running_loss += loss.item()

        
        if i % 200 == 0 or i == n_batches:
            avg = running_loss / i
            print(f"  [Epoch {epoch}]  batch {i:>5}/{n_batches}  "
                  f"loss {avg:.4f}")

    return running_loss / n_batches



@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    for batch in loader:
        src        = batch["src_ids"].to(device)
        dec_input  = batch["dec_input"].to(device)
        dec_target = batch["dec_target"].to(device)

        logits = model(src, dec_input)

        logits     = logits.reshape(-1, logits.size(-1))
        dec_target = dec_target.reshape(-1)

        loss = criterion(logits, dec_target)
        running_loss += loss.item()

    return running_loss / len(loader)



def main():
    cfg    = Config()
    device = get_device()
    print(f"Device: {device}\n")

    
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    vocab_size = tokenizer.vocab_size
    pad_idx    = tokenizer.pad_token_id
    print(f"Tokeniser : {cfg.tokenizer_name}")
    print(f"Vocab size: {vocab_size}")
    print(f"PAD id    : {pad_idx}\n")

    
    train_loader, val_loader = build_loaders(
        tokenizer,
        max_src_len   = cfg.max_enc_len,
        max_tgt_len   = cfg.max_dec_len,
        train_samples = cfg.train_samples,
        batch_size    = cfg.batch_size,
        seed          = cfg.seed,
    )

    
    model = Transformer(
        vocab_size   = vocab_size,
        d_model      = cfg.d_model,
        n_heads      = cfg.n_heads,
        n_enc_layers = cfg.n_enc_layers,
        n_dec_layers = cfg.n_dec_layers,
        d_ff         = cfg.d_ff,
        max_enc_len  = cfg.max_enc_len,
        max_dec_len  = cfg.max_dec_len,
        dropout      = cfg.dropout,
        pad_idx      = pad_idx,
    ).to(device)

    total, trainable = count_parameters(model)
    print(f"\nModel parameters:")
    print(f"  Total     : {total:>12,}")
    print(f"  Trainable : {trainable:>12,}\n")

    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    
    print("=" * 64)
    print(f"  Training for {cfg.epochs} epoch(s)   "
          f"batch_size={cfg.batch_size}  lr={cfg.lr}")
    print("=" * 64)

    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimiser, criterion, device, epoch, cfg.grad_clip
        )
        val_loss = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch}/{cfg.epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"time={elapsed:.0f}s")

        
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch":      epoch,
                    "model":      model.state_dict(),
                    "optimiser":  optimiser.state_dict(),
                    "val_loss":   val_loss,
                    "cfg": {
                        "vocab_size":   vocab_size,
                        "d_model":      cfg.d_model,
                        "n_heads":      cfg.n_heads,
                        "n_enc_layers": cfg.n_enc_layers,
                        "n_dec_layers": cfg.n_dec_layers,
                        "d_ff":         cfg.d_ff,
                        "max_enc_len":  cfg.max_enc_len,
                        "max_dec_len":  cfg.max_dec_len,
                        "dropout":      cfg.dropout,
                        "pad_idx":      pad_idx,
                    },
                },
                cfg.save_path,
            )
            print(f"  ✓ checkpoint saved  (val_loss={val_loss:.4f})")

        print()

    print(f"Training complete.  Best val_loss = {best_val:.4f}\n")

    
    from generate import generate_samples
    generate_samples(model, tokenizer, val_loader, device, n=5)


if __name__ == "__main__":
    main()