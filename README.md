# Text Summarization with Transformer from Scratch

This project implements a complete **Encoder-Decoder Transformer** model from scratch using PyTorch for text summarization. It is trained on the **CNN/DailyMail** dataset and provides a robust framework for sequence-to-sequence learning.

## ✨ Features

- **From-Scratch Transformer Core**:
  - Scaled Dot-Product and Multi-Head Attention.
  - Sinusoidal Positional Encoding.
  - Encoder/Decoder layers with LayerNorm and Residual Connections.
  - Multi-layer Encoder-Decoder stack with cross-attention.
- **CNN/DailyMail Integration**: Automated dataset loading and pre-tokenization via HuggingFace `datasets`.
- **Advanced Decoding**: Supports both **Greedy Search** and **Beam Search** (with length normalization) for high-quality summaries.
- **Training Pipeline**: Comprehensive training loop with validation, gradient clipping, and checkpoint saving.
- **Interactive Mode**: Test the model with your own articles in real-time.

## 📂 Project Structure

- `model.py`: Manual implementation of the Transformer architecture components.
- `dataset.py`: CNN/DailyMail data loading and processing using `bert-base-uncased` tokenization.
- `train.py`: Training script with configurable hyperparameters (`Config` class).
- `generate.py`: Inference utilities, evaluation samples, and interactive summarization demo.

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ and the following packages installed:
```bash
pip install torch transformers datasets
```

### 2. Training
To start training the model on the CNN/DailyMail dataset:
```bash
python train.py
```
*Current configuration (in `train.py`):*
- Epochs: 10
- Batch Size: 16
- Optimizer: AdamW (LR: 3e-4)
- Max Source Length: 512 | Max Target Length: 128

### 3. Inference & Interactive Demo
Use a saved checkpoint (default: `transformer_summ.pt`) to generate summaries for validation samples or your own text:
```bash
python generate.py
```
This script will:
1. Load the best saved model.
2. Print sample summaries (Greedy & Beam Search).
3. Enter an interactive loop where you can paste an article and get a generated summary.

## 🛠 Model Configuration

The `Config` class in `train.py` allows easy experimentation:
- `d_model`: 256 (Hidden dimension)
- `n_heads`: 4 (Attention heads)
- `n_enc_layers`/`n_dec_layers`: 4 layers each
- `d_ff`: 1024 (Feed-forward network dimension)

## 📊 Performance
The model is trained to minimize CrossEntropyLoss on the CNN/DailyMail highlights. Beam search (default `beam_width=4`) significantly improves summary quality by exploring multiple hypotheses.

---
*Built with PyTorch. References: "Attention Is All You Need" (Vaswani et al., 2017).*
