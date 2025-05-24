# Neural Memory Transformer

A PyTorch implementation of a Transformer model enhanced with a Neural Memory Module inspired by the Titans architecture. This model incorporates Long-Term Memory (LMM) capabilities that can dynamically update during both training and inference, enabling the model to adapt and learn from sequences in real-time.

## Features

- **Neural Memory Module (LMM)**: A learnable memory system that updates its parameters based on prediction errors
- **Modern Transformer Architecture**: Includes RMSNorm, Rotary Positional Embeddings (RoPE), and SwiGLU activation
- **Dynamic Memory Updates**: Memory parameters can be updated during inference for continual learning
- **BPE Tokenization**: Custom tokenizer wrapper with streaming data support
- **Efficient Training**: Streaming dataset implementation for large text corpora

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- tqdm
- tokenizers

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Training a Model

1. Place your text data in the `smalldata/` directory (sample educational texts are included)

2. Run the training script:
```bash
python train_model.py
```

The script will:
- Train a BPE tokenizer on your data
- Create a streaming dataloader
- Train the Neural Memory Transformer
- Save checkpoints and generate sample text during training

### Configuration

The model configuration can be modified in `train_model.py`. Key parameters include:

```python
config = NeuralMemoryTransformerConfig(
    vocab_size=5000,
    d_model=512,
    n_layers=6,
    n_heads=8,
    ffn_dim=2048,
    max_seq_len=128,
    
    # Neural Memory specific
    memory_dim=512,
    lmm_layers=2,
    lmm_learning_rate=0.01,
    lmm_momentum_decay=0.9,
    lmm_weight_decay=0.01,
    lmm_update_loss_threshold=0.1,
    update_lmm_at_test_time=False
)
```

## Architecture

### Neural Memory Module

The Neural Memory Module (LMM) is the key innovation, implementing a learnable memory system that:
- Maintains its own parameters (keys, values, queries)
- Updates based on prediction errors using momentum-based optimization
- Can continue learning during inference if enabled

### Model Components

1. **Embedding Layer**: Token embeddings with positional encoding
2. **Transformer Blocks**: Each containing:
   - Multi-head attention with RoPE
   - Neural Memory Module
   - Feed-forward network with SwiGLU activation
   - RMSNorm for layer normalization
3. **Output Layer**: Linear projection to vocabulary

## File Structure

```
super-sercret-memory-transformer/
├── neural_memory_transformer_model.py  # Model implementation
├── train_model.py                      # Training script
├── dataloader.py                       # Data loading and tokenization
├── requirements.txt                    # Dependencies
├── smalldata/                          # Sample training data
│   ├── 100kmath.txt
│   ├── A First Course in Linear Algebra Robert A. Beezer.txt
│   ├── NEW EDITION HIGH SCHOOL English Grammar & Composition BY WREN & MARTIN.txt
│   └── U.S. History SENIOR CONTRIBUTING AUTHORS P. SCOTT CORBETT, VENTURA COLLEGE.txt
└── output_titans_inspired_transformer/ # Training outputs (created during training)
```

## Memory Statistics

During training, the model provides memory statistics including:
- Average memory activation
- Memory update rate
- Memory gradient norms
- Memory capacity utilization

These can be accessed via:
```python
memory_stats = model.get_memory_stats()
```

## Customization

### Adjusting Memory Behavior

- **`lmm_update_loss_threshold`**: Controls when memory updates occur (based on prediction error)
- **`update_lmm_at_test_time`**: Enable/disable memory updates during inference
- **`lmm_learning_rate`**: Controls how quickly memory adapts
- **`lmm_momentum_decay`**: Momentum for memory parameter updates

### Adding Custom Data

1. Add `.txt` files to the `smalldata/` directory
2. The tokenizer will automatically include them during training
3. Adjust `vocab_size` in the config if needed

## Training Details

The training script includes:
- Automatic mixed precision (AMP) support
- Gradient accumulation
- Learning rate scheduling with warmup
- Periodic checkpoint saving
- Sample text generation during training
- Memory statistics logging

## Citation

This implementation is inspired by the Titans architecture for neural memory in transformers. The key innovation is the integration of a learnable memory module that can adapt during both training and inference.
