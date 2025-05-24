import torch
import torch.nn as nn
import math
from dataclasses import dataclass, field


@dataclass
class StandardTransformerConfig:
    """Configuration for the Standard Transformer Language Model."""

    vocab_size: int = 5000  # Default, will be updated by tokenizer
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    ffn_dim: int = 2048  # Typically 4 * d_model
    dropout: float = 0.1
    max_seq_len: int = 128  # Default, can be overridden by dataloader's max_seq_len
    pad_idx: int = 0  # Default, will be updated by tokenizer
    norm_first: bool = True  # For Pre-LayerNorm
    activation_function: str = "gelu"  # "relu" or "gelu"

    def __post_init__(self):
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.d_model


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] if batch_first=False
               or [batch_size, seq_len, embedding_dim] if batch_first=True
        """
        # Assuming x is [batch_size, seq_len, embedding_dim]
        x = x + self.pe[: x.size(1)].transpose(0, 1)
        return self.dropout(x)


class StandardTransformerLM(nn.Module):
    def __init__(self, config: StandardTransformerConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_idx
        )
        self.pos_encoder = PositionalEncoding(
            config.d_model, config.dropout, config.max_seq_len
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation=config.activation_function,
            batch_first=True,  # Important: input tensors are (batch, seq, feature)
            norm_first=config.norm_first,  # Pre-LayerNorm
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            norm=(
                nn.LayerNorm(config.d_model) if config.norm_first else None
            ),  # Final norm if Pre-LN
        )
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)

        self._init_weights()

    def _init_weights(self):
        # Initialize embedding layer and output projection
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

        # Initialize Transformer layers (often handled by PyTorch default, but can be customized)
        for layer in self.transformer_encoder.layers:
            # For Linear layers in MHA and FFN
            if hasattr(layer, "self_attn"):  # MultiheadAttention
                nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
                if layer.self_attn.in_proj_bias is not None:
                    nn.init.zeros_(layer.self_attn.in_proj_bias)
                nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
                if layer.self_attn.out_proj.bias is not None:
                    nn.init.zeros_(layer.self_attn.out_proj.bias)

            if hasattr(layer, "linear1"):  # FFN
                nn.init.xavier_uniform_(layer.linear1.weight)
                if layer.linear1.bias is not None:
                    nn.init.zeros_(layer.linear1.bias)
            if hasattr(layer, "linear2"):  # FFN
                nn.init.xavier_uniform_(layer.linear2.weight)
                if layer.linear2.bias is not None:
                    nn.init.zeros_(layer.linear2.bias)

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generates a causal mask for autoregressive decoding."""
        mask = (torch.triu(torch.ones(size, size, device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(
        self, input_ids: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            padding_mask: Tensor of shape (batch_size, seq_len) where True indicates a pad token.
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 1. Token Embeddings
        embeds = self.token_embedding(input_ids) * math.sqrt(
            self.config.d_model
        )  # Scale embedding

        # 2. Positional Encoding
        embeds = self.pos_encoder(embeds)  # expects (batch, seq, feature)

        # 3. Causal Mask for self-attention
        # Needs to be (seq_len, seq_len) for TransformerEncoder
        causal_mask = self._generate_causal_mask(seq_len, device)

        # 4. Transformer Encoder (used as a decoder-only stack)
        # src_key_padding_mask should be (batch_size, seq_len)
        transformer_output = self.transformer_encoder(
            src=embeds, mask=causal_mask, src_key_padding_mask=padding_mask
        )

        # 5. Output Projection
        logits = self.output_projection(transformer_output)
        return logits

    def get_config_dict(self) -> dict:
        return self.config.__dict__
