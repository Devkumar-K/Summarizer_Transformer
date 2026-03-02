# CNN/DailyMail Text Summarization with Transformer (from scratch)

A from-scratch implementation of the **Transformer** (Vaswani et al., 2017) for abstractive text summarization on the **CNN/DailyMail** dataset.

This project includes:

- Full Transformer encoder-decoder architecture implemented in PyTorch
- Custom dataset preparation for summarization
- Training loop with gradient clipping and checkpointing
- Greedy + Beam search decoding (with length penalty)
- Interactive summarization demo

## Features

- Pure PyTorch (no `nn.Transformer` usage)
- BERT tokenizer (`bert-base-uncased`)
- Positional encoding, multi-head attention, feed-forward layers
- Proper causal + padding masking
- Beam search with length penalty
- Simple model checkpointing & best-model selection
- Sample generation on validation set (greedy + beam)

## Project Structure
├── dataset.py       # Dataset + DataLoader for CNN/DailyMail
├── model.py         # Full Transformer implementation (Encoder + Decoder)
├── train.py         # Training loop + validation
├── generate.py      # Inference: greedy + beam search + interactive mode
├── transformer_summ.pt   # (generated) best checkpoint
└── README.md

## Requirements


torch>=2.0
transformers>=4.30
datasets>=2.14
torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121   # or cu118 / cpu
pip install transformers datasets##

Quick Start

1.Train the model
python train.py

Default config (can be changed in Config class):

*40,000 training examples
*batch size 16
*d_model=256, 4 heads, 4 enc/dec layers
*~14–15M parameters
*10 epochs

2.Generate summaries (after training)
python generate.py

What it does:
Loads the best checkpoint (transformer_summ.pt)
Shows greedy + beam search (width=4) summaries on 5 validation examples
Starts interactive mode — paste any article and get a summary

Model Configuration (default)

d_model = 256
heads = 4
encoder layers = 4
decoder layers = 4
feed-forward dim = 1024
dropout = 0.1
max src len = 512
max tgt len = 128
learning rate = 3e-4
optimizer = AdamW
gradient clip = 1.0

Example output style:
Ground truth:
  Police are searching for a man who robbed a bank in broad daylight...

Predicted (greedy):
  Authorities are looking for a suspect after a bank robbery occurred...

Predicted (beam):
  A man robbed a bank in the middle of the day and fled the scene, police said.

Acknowledgments:

  *Dataset: CNN/DailyMail 3.0
  *Architecture inspired by: "Attention is All You Need" (Vaswani et al., 2017)
  *Hugging Face transformers and datasets libraries
