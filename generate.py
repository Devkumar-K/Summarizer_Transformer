import torch
from transformers import AutoTokenizer

from model   import Transformer, count_parameters
from dataset import build_loaders





@torch.no_grad()
def greedy_decode(model, src_ids, tokenizer, max_len=128, device="cpu"):
   
    model.eval()

    bos = tokenizer.cls_token_id
    eos = tokenizer.sep_token_id

    
    if src_ids.dim() == 1:
        src_ids = src_ids.unsqueeze(0)
    src_ids = src_ids.to(device)

    
    src_mask = model.make_src_mask(src_ids)             
    enc_out  = model.encode(src_ids, src_mask)         

    
    dec_ids = torch.tensor([[bos]], dtype=torch.long, device=device)

    generated = []

    for _ in range(max_len - 1):
        tgt_mask   = model.make_tgt_mask(dec_ids)       
        cross_mask = model.make_src_mask(src_ids)       

        dec_out = model.decode(dec_ids, enc_out,
                               tgt_mask, cross_mask)    
        logits  = model.project(dec_out[:, -1, :])      

        next_tok = logits.argmax(dim=-1).item()

        if next_tok == eos:
            break

        generated.append(next_tok)

        dec_ids = torch.cat(
            [dec_ids,
             torch.tensor([[next_tok]], dtype=torch.long, device=device)],
            dim=1,
        )

    return tokenizer.decode(generated, skip_special_tokens=True)





@torch.no_grad()
def beam_search_decode(
    model,
    src_ids,
    tokenizer,
    beam_width: int = 4,
    max_len: int = 128,
    length_penalty: float = 0.6,
    device: str = "cpu",
):
    
    model.eval()

    bos = tokenizer.cls_token_id
    eos = tokenizer.sep_token_id

    if src_ids.dim() == 1:
        src_ids = src_ids.unsqueeze(0)
    src_ids = src_ids.to(device)

    
    src_mask = model.make_src_mask(src_ids)
    enc_out  = model.encode(src_ids, src_mask)

    
    beams = [{"tokens": [bos], "log_prob": 0.0, "done": False}]

    def _score(b):
        
        length = max(len(b["tokens"]) - 1, 1)     
        return b["log_prob"] / (length ** length_penalty)

    for _ in range(max_len - 1):
        candidates = []

        for beam in beams:
           
            if beam["done"]:
                candidates.append(beam)
                continue

            
            dec_ids    = torch.tensor(
                [beam["tokens"]], dtype=torch.long, device=device
            )
            tgt_mask   = model.make_tgt_mask(dec_ids)
            cross_mask = model.make_src_mask(src_ids)

            dec_out = model.decode(dec_ids, enc_out, tgt_mask, cross_mask)
            logits  = model.project(dec_out[:, -1, :])        
            log_p   = torch.log_softmax(logits, dim=-1).squeeze(0)

            
            topk_lp, topk_id = log_p.topk(beam_width)

            for k in range(beam_width):
                tok = topk_id[k].item()
                candidates.append({
                    "tokens":   beam["tokens"] + [tok],
                    "log_prob": beam["log_prob"] + topk_lp[k].item(),
                    "done":     tok == eos,
                })

        
        candidates.sort(key=_score, reverse=True)
        beams = candidates[:beam_width]

        if all(b["done"] for b in beams):
            break

    
    best = max(beams, key=_score)
    tokens = best["tokens"][1:]                     
    if tokens and tokens[-1] == eos:
        tokens = tokens[:-1]                        

    return tokenizer.decode(tokens, skip_special_tokens=True)



def summarize_text(
    model,
    tokenizer,
    text: str,
    max_src_len: int = 512,
    max_tgt_len: int = 128,
    beam_width: int = 1,
    device: str = "cpu",
):
    """
    Tokenise a raw article string, then generate its summary.
    beam_width=1 → greedy, >1 → beam search.
    """
    src_tokens = tokenizer.encode(text, add_special_tokens=False)[:max_src_len]
    src_ids    = torch.tensor(src_tokens, dtype=torch.long)

    if beam_width > 1:
        return beam_search_decode(
            model, src_ids, tokenizer,
            beam_width=beam_width, max_len=max_tgt_len, device=device,
        )
    return greedy_decode(model, src_ids, tokenizer,
                         max_len=max_tgt_len, device=device)



def generate_samples(model, tokenizer, val_loader, device, n=5):
    """
    Pick the first `n` validation samples, generate summaries with
    both greedy and (optionally) beam search, and print everything.
    """
    model.eval()
    ds = val_loader.dataset

    print("=" * 80)
    print("  SAMPLE SUMMARIES  (greedy decoding)")
    print("=" * 80)

    for i in range(min(n, len(ds))):
        sample   = ds[i]
        src_ids  = sample["src_ids"]
        tgt_ids  = sample["dec_target"]

        
        gt_tokens = [
            t.item() for t in tgt_ids
            if t.item() not in (tokenizer.pad_token_id, tokenizer.sep_token_id)
        ]
        ground_truth = tokenizer.decode(gt_tokens, skip_special_tokens=True)

        
        src_tokens = [t.item() for t in src_ids if t.item() != tokenizer.pad_token_id]
        source_text = tokenizer.decode(src_tokens, skip_special_tokens=True)

        
        pred_greedy = greedy_decode(model, src_ids, tokenizer, device=device)

        print(f"\n--- Sample {i+1} ---")
        print(f"Article (first 500 chars):\n  {source_text[:500]}…\n")
        print(f"Ground truth:\n  {ground_truth}\n")
        print(f"Predicted (greedy):\n  {pred_greedy}")
        print("-" * 80)

    
    print("\n" + "=" * 80)
    print("  BEAM SEARCH  (beam_width=4)")
    print("=" * 80)
    for i in range(min(3, len(ds))):
        sample  = ds[i]
        src_ids = sample["src_ids"]

        pred_beam = beam_search_decode(
            model, src_ids, tokenizer,
            beam_width=4, device=device,
        )

        gt_tokens = [
            t.item() for t in sample["dec_target"]
            if t.item() not in (tokenizer.pad_token_id, tokenizer.sep_token_id)
        ]
        ground_truth = tokenizer.decode(gt_tokens, skip_special_tokens=True)

        print(f"\n--- Sample {i+1} ---")
        print(f"Ground truth:\n  {ground_truth}\n")
        print(f"Predicted (beam):\n  {pred_beam}")
        print("-" * 80)



def main():
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    
    ckpt_path = "transformer_summ.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt["cfg"]
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(val_loss={ckpt['val_loss']:.4f})")

    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    
    model = Transformer(
        vocab_size   = c["vocab_size"],
        d_model      = c["d_model"],
        n_heads      = c["n_heads"],
        n_enc_layers = c["n_enc_layers"],
        n_dec_layers = c["n_dec_layers"],
        d_ff         = c["d_ff"],
        max_enc_len  = c["max_enc_len"],
        max_dec_len  = c["max_dec_len"],
        dropout      = c["dropout"],
        pad_idx      = c["pad_idx"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    total, trainable = count_parameters(model)
    print(f"Parameters: {total:,} total, {trainable:,} trainable\n")

    # ---- load validation data ----
    _, val_loader = build_loaders(
        tokenizer,
        max_src_len   = c["max_enc_len"],
        max_tgt_len   = c["max_dec_len"],
        batch_size    = 1,
        num_workers   = 0,
    )

    # ---- generate samples ----
    generate_samples(model, tokenizer, val_loader, device, n=5)

    # ---- interactive demo ----
    print("\n" + "=" * 80)
    print("  Interactive mode  (type an article, get a summary)")
    print("  Type 'quit' to exit.")
    print("=" * 80)
    while True:
        text = input("\nArticle >>> ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue
        summary = summarize_text(
            model, tokenizer, text,
            max_src_len=c["max_enc_len"],
            max_tgt_len=c["max_dec_len"],
            beam_width=4,
            device=device,
        )
        print(f"\nSummary: {summary}")


if __name__ == "__main__":
    main()