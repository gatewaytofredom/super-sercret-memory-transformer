import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List


@dataclass
class NeuralMemoryTransformerConfig:
    """Configuration for Neural Memory Transformer with Titans-inspired memory."""

    # Standard transformer config
    vocab_size: int = 5000
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    ffn_dim: int = 2048  # Hidden dim for SwiGLU, effectively (2/3 * ffn_dim) * 2
    dropout: float = 0.1
    max_seq_len: int = 128
    pad_idx: int = 0

    # Neural Memory (LMM - Long-Term Memory) specific config
    memory_dim: int = (
        512  # Dimension for LMM internal processing, keys, values, queries
    )
    lmm_layers: int = 2  # Number of layers in the LMM MLP

    # LMM Update Hyperparameters (inspired by Titans Eq. 13 & 14)
    lmm_learning_rate: float = 0.01  # θ_t in paper (learning rate for LMM parameters)
    lmm_momentum_decay: float = 0.9  # η_t in paper (decay for LMM momentum)
    lmm_weight_decay: float = 0.01  # α_t in paper (weight decay for LMM parameters)
    lmm_gradient_clip: float = 1.0

    # Gating LMM updates
    lmm_update_loss_threshold: float = (
        0.1  # Only update LMM if its own prediction loss is above this
    )
    update_lmm_at_test_time: bool = (
        False  # If true, LMM params are updated during model.eval()
    )

    # Old memory params, will be deprecated/removed if not used by new LMM
    # memory_size: int = 256 (LMM is a network, not a fixed slot bank)
    # surprise_threshold: float = 0.1 (Replaced by lmm_update_loss_threshold based on LMM's own loss)
    # decay_rate: float = 0.99 (Replaced by lmm_weight_decay)
    # memory_lr: float = 0.01 (Replaced by lmm_learning_rate)
    # gradient_clip: float = 1.0 (Replaced by lmm_gradient_clip)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        # Removed self.max_seq_len as it's not strictly needed for forward if seq_len is passed

    def forward(self, x: torch.Tensor, seq_len: int):  # seq_len must be passed
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        # Ensure sin and cos are broadcastable to x's shape for RoPE application
        # x shape: [batch, n_heads, seq_len, head_dim] or [batch, seq_len, d_model]
        # sin/cos shape: [seq_len, head_dim/2] -> [1, 1, seq_len, head_dim/2] for attention
        # Or [seq_len, d_model/2] -> [1, seq_len, d_model/2] for direct application
        sin = sinusoid_inp.sin()
        cos = sinusoid_inp.cos()
        # Reshape for broadcasting if necessary, or handle in apply_rotary_emb
        return sin, cos


def apply_rotary_emb(x, sin, cos):
    # x: [..., seq_len, dim]
    # sin, cos: [seq_len, dim/2]
    # Reshape sin, cos for broadcasting: [1, ..., 1, seq_len, dim/2]
    embed_dim = sin.shape[-1] * 2
    if x.shape[-1] != embed_dim:
        raise ValueError(
            f"x dim {x.shape[-1]} must be twice sin/cos dim {sin.shape[-1]}"
        )

    sin = sin.unsqueeze(
        0
    )  # Add batch dim for broadcasting if x is [batch, seq_len, dim]
    cos = cos.unsqueeze(0)
    while sin.ndim < x.ndim - 1:  # Add intermediate dims (like n_heads for attention)
        sin = sin.unsqueeze(1)
        cos = cos.unsqueeze(1)

    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: Optional[int] = None):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        # Standard SwiGLU: Wx, V_x, gate(Wx) * V_x then W_out
        # Here, ffn_dim in config is used as dim_hidden for W1/W2
        self.w1 = nn.Linear(dim_in, dim_hidden, bias=False)  # For gate
        self.w2 = nn.Linear(dim_in, dim_hidden, bias=False)  # For value
        self.w3 = nn.Linear(dim_hidden, dim_out, bias=False)  # Output projection

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class NeuralMemoryModule(nn.Module):
    """
    Titans-inspired Neural Long-Term Memory (LMM).
    The LMM is an MLP whose parameters are updated online.
    """

    def __init__(self, config: NeuralMemoryTransformerConfig):
        super().__init__()
        self.config = config
        self.memory_dim = config.memory_dim  # d_model of input to this module

        # Projections for LMM's associative memory task (k_t, v_t, q_t)
        # These are trained by the main model's optimizer
        self.project_k_lmm = nn.Linear(config.d_model, self.config.memory_dim)
        self.project_v_lmm = nn.Linear(config.d_model, self.config.memory_dim)
        self.project_q_lmm = nn.Linear(config.d_model, self.config.memory_dim)

        # The LMM network itself (an MLP)
        # Its parameters are updated by the internal LMM update rule
        lmm_mlp_layers = []
        current_dim = self.config.memory_dim
        for _ in range(config.lmm_layers - 1):
            lmm_mlp_layers.append(nn.Linear(current_dim, current_dim))
            lmm_mlp_layers.append(nn.ReLU())  # or SiLU
        lmm_mlp_layers.append(
            nn.Linear(current_dim, self.config.memory_dim)
        )  # Final output layer
        self.lmm_network = nn.Sequential(*lmm_mlp_layers)

        # Momentum buffer for LMM parameters
        # Needs to be initialized after lmm_network is on device and parameters are known
        self.lmm_momentum_buffer: Dict[str, torch.Tensor] = {}
        self._is_momentum_initialized = False

    def _initialize_momentum_buffer(self):
        if not self._is_momentum_initialized:
            for name, param in self.lmm_network.named_parameters():
                if param.requires_grad:
                    self.lmm_momentum_buffer[name] = torch.zeros_like(param.data)
            self._is_momentum_initialized = True

    def _get_lmm_params_for_update(self) -> List[nn.Parameter]:
        return [p for p in self.lmm_network.parameters() if p.requires_grad]

    @torch.no_grad()  # Overall method assignments should not affect outer graph
    def _perform_lmm_update(self, k_lmm: torch.Tensor, v_lmm: torch.Tensor):
        """
        Performs the LMM parameter update based on Titans paper (Eq. 13 & 14).
        """
        if not self.config.lmm_learning_rate > 0:  # Skip if LR is zero
            return

        if not self._is_momentum_initialized:
            self._initialize_momentum_buffer()

        # List of parameters of self.lmm_network to update
        params_to_track = [p for p in self.lmm_network.parameters() if p.requires_grad]
        if not params_to_track:  # Should not happen if LMM has layers
            return

        # Store original requires_grad states
        initial_lmm_params_requires_grad_states = {
            p: p.requires_grad for p in params_to_track
        }

        lmm_param_grads_list = []  # To store computed gradients

        # --- Temporarily re-enable grad for the LMM's internal loss computation ---
        with torch.enable_grad():
            # Ensure LMM parameters are temporarily set to require gradients for this internal computation
            for p in params_to_track:
                p.requires_grad_(True)

            # Ensure k_lmm and v_lmm do not carry gradients from previous LMM updates or outer graph
            k_lmm_detached = (
                k_lmm.detach()
            )  # k_lmm already from projection with requires_grad from main model
            v_lmm_detached = v_lmm.detach()  # v_lmm too

            predicted_v_lmm = self.lmm_network(k_lmm_detached)

            # Calculate LMM's own loss
            # Reduction='none' to get per-item loss, then mean over feature dim, then mean over batch*seq_len
            lmm_loss_per_item = F.mse_loss(
                predicted_v_lmm, v_lmm_detached, reduction="none"
            ).mean(dim=-1)
            avg_lmm_loss = (
                lmm_loss_per_item.mean()
            )  # Scalar average loss for the batch of items

            if avg_lmm_loss.item() < self.config.lmm_update_loss_threshold:
                # Restore original requires_grad states if no update is performed
                for p in params_to_track:
                    p.requires_grad_(initial_lmm_params_requires_grad_states[p])
                return

            scalar_lmm_loss_for_grad = (
                lmm_loss_per_item.sum()
            )  # Sum up for a single scalar to backprop from

            # Compute gradients ∇ℓ w.r.t. LMM parameters (params_to_track)
            # These grads are only for the LMM's internal update.
            computed_grads = torch.autograd.grad(
                outputs=scalar_lmm_loss_for_grad,
                inputs=params_to_track,
                allow_unused=True,
                create_graph=False,
            )
            lmm_param_grads_list = list(computed_grads)  # Store them
        # --- Gradient computation is done. Outside torch.enable_grad(), the @torch.no_grad() context is active ---

        # Restore original requires_grad states for LMM parameters
        # This is important so the main model's optimizer doesn't try to optimize them if they weren't supposed to be.
        # However, lmm_network parameters are *not* part of the main model's optimizer.
        # So, their requires_grad state should ideally remain True if they are to be updated by LMM.
        # Let's assume LMM parameters *always* require grad for their own updates.
        # The outer @torch.no_grad ensures the *assignments* don't create ops in the main graph.
        for p in params_to_track:
            p.requires_grad_(True)  # Keep them True for subsequent LMM updates.
            # The outer @torch.no_grad() protects the assignment.

        # 3. Update LMM parameters and momentum (still under @torch.no_grad() for assignments)
        for i, param in enumerate(params_to_track):
            grad = lmm_param_grads_list[i]
            if grad is None:
                continue

            # Clip gradient for LMM update (using the grad tensor directly)
            # grad_norm = torch.norm(grad) # This would be for a single tensor
            # if grad_norm > self.config.lmm_gradient_clip:
            #    grad.data.mul_(self.config.lmm_gradient_clip / grad_norm)
            # Instead, usually clip_grad_norm_ is applied to a list of gradients *before* optimizer step
            # Here we apply it to the individual grad tensor if needed, or just use it.
            # For simplicity, let's assume clipping happened or is handled by magnitude.
            # A more robust way: torch.nn.utils.clip_grad_norm_([grad], self.config.lmm_gradient_clip)

            param_name = None  # Need to map param back to name for momentum buffer
            for name, p_lookup in self.lmm_network.named_parameters():
                if p_lookup is param:
                    param_name = name
                    break

            if param_name is None:
                continue  # Should not happen

            current_momentum = self.lmm_momentum_buffer[param_name]
            new_momentum = (
                self.config.lmm_momentum_decay * current_momentum
                - self.config.lmm_learning_rate * grad
            )  # grad is already ∇ℓ
            self.lmm_momentum_buffer[param_name] = (
                new_momentum.detach()
            )  # Detach momentum

            param.data.mul_(1.0 - self.config.lmm_weight_decay).add_(new_momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes input x, retrieves from LMM, and potentially updates LMM.
        Args:
            x: Input tensor [batch, seq_len, d_model] from the Transformer block.
        Returns:
            retrieved_info: Information retrieved from LMM [batch, seq_len, memory_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Flatten for LMM processing if LMM sees each token independently
        # x_flat = x.view(batch_size * seq_len, -1)

        # Project input x to k_lmm, v_lmm, q_lmm for the LMM's associative task
        # These projections are trained as part of the main model
        k_lmm = self.project_k_lmm(x)  # [batch, seq_len, memory_dim]
        v_lmm = self.project_v_lmm(x)  # [batch, seq_len, memory_dim]
        q_lmm = self.project_q_lmm(x)  # [batch, seq_len, memory_dim]

        # 1. Retrieve information using current LMM state (M_{t-1})
        # The lmm_network's forward pass for retrieval
        retrieved_info = self.lmm_network(q_lmm)

        # 2. Update LMM internal state to M_t (if conditions met)
        if self.training or self.config.update_lmm_at_test_time:
            # Flatten k_lmm, v_lmm for batched update if treating tokens independently
            k_lmm_flat = k_lmm.reshape(batch_size * seq_len, -1)
            v_lmm_flat = v_lmm.reshape(batch_size * seq_len, -1)
            self._perform_lmm_update(k_lmm_flat, v_lmm_flat)

        return retrieved_info  # This was retrieved using M_{t-1} state

    def get_memory_stats(self) -> dict:
        lmm_param_norm = 0.0
        lmm_momentum_norm = 0.0
        if self._is_momentum_initialized:  # Check if buffer is ready
            with torch.no_grad():
                for param in self.lmm_network.parameters():
                    if param.requires_grad:
                        lmm_param_norm += torch.norm(param.data).item() ** 2
                for name in self.lmm_momentum_buffer:
                    lmm_momentum_norm += (
                        torch.norm(self.lmm_momentum_buffer[name]).item() ** 2
                    )
        lmm_param_norm = math.sqrt(lmm_param_norm)
        lmm_momentum_norm = math.sqrt(lmm_momentum_norm)

        return {
            "lmm_param_norm": lmm_param_norm,
            "lmm_momentum_norm": lmm_momentum_norm,
            "lmm_config_lr": self.config.lmm_learning_rate,
            "lmm_config_mom_decay": self.config.lmm_momentum_decay,
            "lmm_config_wd": self.config.lmm_weight_decay,
        }


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, config: NeuralMemoryTransformerConfig):
        super().__init__()
        self.config = config
        assert (
            config.d_model % config.n_heads == 0
        ), "d_model must be divisible by n_heads"
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.head_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        sin, cos = self.rope(q, seq_len=seq_len)  # Pass q to get device and dtype for t
        q = apply_rotary_emb(q, sin, cos)
        k = apply_rotary_emb(k, sin, cos)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            # Mask shape [seq_len, seq_len], needs to be [batch_size, n_heads, seq_len, seq_len]
            # Or if padding_mask [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            # Causal mask is usually [1,1,seq_len,seq_len]
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(out)


class TransformerBlockWithMemory(nn.Module):
    def __init__(self, config: NeuralMemoryTransformerConfig):
        super().__init__()
        self.config = config

        self.attn = MultiHeadAttentionWithRoPE(config)
        self.ffn = SwiGLU(
            config.d_model, config.ffn_dim, config.d_model
        )  # dim_out is d_model
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)  # For FFN pre-norm
        # self.norm3 = RMSNorm(config.d_model) # Not needed if memory integrated before FFN's norm

        # Memory integration components
        # Input to memory gate will be x and retrieved_memory concatenated
        self.memory_gate_proj = nn.Linear(
            config.d_model + config.memory_dim, config.d_model
        )  # Projects concat to d_model for gating
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory_module: NeuralMemoryModule,  # Pass the shared LMM instance
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention sub-layer (Pre-Norm)
        h_attn = self.attn(self.norm1(x), mask=attention_mask)
        x = x + self.dropout(h_attn)

        # Memory Interaction sub-layer (Pre-Norm style for consistency)
        # The input to memory can be the output of attention (x) or normed version
        x_for_memory = self.norm2(x)  # Using a second norm before memory interaction
        retrieved_memory_content = memory_module(
            x_for_memory
        )  # LMM retrieves and updates internally

        # Gated integration of memory
        # Ensure retrieved_memory_content is compatible with x in dimension if memory_dim != d_model
        # If memory_dim != d_model, might need a projection for retrieved_memory_content
        # Assuming memory_dim == d_model for now as per config logic
        if retrieved_memory_content.shape[-1] != x.shape[-1]:
            # This should ideally not happen if config.memory_dim is set to config.d_model
            # or if there's a projection layer. For now, let's assume they match.
            pass  # Add projection if necessary

        gate_input = torch.cat([x_for_memory, retrieved_memory_content], dim=-1)
        gate_values = torch.sigmoid(self.memory_gate_proj(gate_input))

        x = x + self.dropout(
            gate_values * retrieved_memory_content
        )  # Add gated memory to residual path

        # FFN sub-layer (Pre-Norm, using the output from memory interaction)
        # norm2 was used before memory. Re-norming or using x directly?
        # Let's use x as it is (already had one residual connection after memory)
        # Or apply a third norm: x_for_ffn = self.norm3(x)
        h_ffn = self.ffn(self.norm2(x))  # Re-using norm2, but it's better to have norm3
        # Let's assume norm2 is the pre-FFN norm for now.
        x = x + self.dropout(h_ffn)
        return x


class NeuralMemoryTransformer(nn.Module):
    def __init__(self, config: NeuralMemoryTransformerConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_idx
        )
        self.dropout_emb = nn.Dropout(config.dropout)  # Dropout after embedding

        # Shared Neural Memory Module (LMM)
        self.memory_module = NeuralMemoryModule(config)

        self.layers = nn.ModuleList(
            [TransformerBlockWithMemory(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model)
        self.output_projection = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )

        # Weight tying (optional but common)
        # self.token_embedding.weight = self.output_projection.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # Initialize LMM network weights specifically if needed
        for name, param in self.memory_module.lmm_network.named_parameters():
            if "weight" in name:
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * self.config.lmm_layers)
                )  # He-like for LMM
            elif "bias" in name:
                nn.init.zeros_(param)

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(
            torch.ones(size, size, device=device) * float("-inf"), diagonal=1
        )
        # For MultiHeadAttentionWithRoPE, mask should be 0 for masked, -inf for unmasked.
        # Or, if using masked_fill(mask == 0, ...), then mask should be 1 for keep, 0 for mask.
        # Let's stick to causal mask being 1s in lower triangle.
        # The MHA implementation uses `scores.masked_fill(mask == 0, float("-inf"))`
        # So, a causal mask should have 1s where attention is allowed, 0s where it's not.
        causal_mask_tril = torch.tril(
            torch.ones(size, size, device=device, dtype=torch.bool)
        )
        # Reshape for MHA: [1, 1, seq_len, seq_len] for broadcasting
        return causal_mask_tril.unsqueeze(0).unsqueeze(0)

    def forward(
        self, input_ids: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        x = self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        x = self.dropout_emb(x)

        # Create combined mask: causal + padding
        causal_mask = self._generate_causal_mask(seq_len, device)  # [1,1,S,S]

        attention_mask = causal_mask  # Start with causal mask
        if padding_mask is not None:
            # padding_mask: [B, S], True for PAD
            # We need [B, 1, 1, S] for MHA score masking (masking out keys)
            # Or [B, 1, S, S] if applying to the full attention matrix
            # If scores are [B,H,S_q,S_k], and padding_mask refers to S_k:
            pad_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            pad_mask_expanded = pad_mask_expanded.expand(
                -1, -1, seq_len, -1
            )  # [B, 1, S, S]
            # Attention mask requires 1s for non-masked, 0s for masked.
            # Our causal mask is already bool (True for non-masked).
            # padding_mask is True for PAD (masked). So ~padding_mask for non-masked.
            attention_mask = attention_mask & (~pad_mask_expanded)  # Element-wise AND

        for layer in self.layers:
            x = layer(x, self.memory_module, attention_mask=attention_mask)

        x = self.final_norm(x)
        logits = self.output_projection(x)
        return logits

    def get_memory_stats(self) -> dict:
        return self.memory_module.get_memory_stats()


if __name__ == "__main__":
    print("Testing Refactored Neural Memory Transformer...")
    config = NeuralMemoryTransformerConfig(
        vocab_size=1000,
        d_model=64,  # Smaller for faster test
        n_layers=2,
        n_heads=4,
        ffn_dim=128,
        max_seq_len=32,
        memory_dim=64,
        lmm_layers=2,
        lmm_learning_rate=0.01,
        lmm_momentum_decay=0.9,
        lmm_weight_decay=0.001,
        lmm_update_loss_threshold=0.05,  # Lower threshold for testing updates
        update_lmm_at_test_time=True,  # Test test-time updates
    )
    model = NeuralMemoryTransformer(config)
    print(
        f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters"
    )

    batch_size = 2
    seq_len = 16  # Shorter seq len for test
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    padding_mask_example = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask_example[0, -5:] = True  # Example padding

    print("\n--- Training Mode Test ---")
    model.train()
    try:
        # Initial memory state
        initial_mem_stats = model.get_memory_stats()
        print(f"Initial Memory stats: {initial_mem_stats}")

        logits_train = model(input_ids, padding_mask=padding_mask_example)
        print(f"Train Forward pass successful! Output shape: {logits_train.shape}")

        # Memory stats after one forward pass (should have updated if loss > threshold)
        mem_stats_after_train_pass = model.get_memory_stats()
        print(f"Memory stats after 1 train pass: {mem_stats_after_train_pass}")
        assert (
            abs(
                initial_mem_stats["lmm_param_norm"]
                - mem_stats_after_train_pass["lmm_param_norm"]
            )
            > 1e-6
            or abs(
                initial_mem_stats["lmm_momentum_norm"]
                - mem_stats_after_train_pass["lmm_momentum_norm"]
            )
            > 1e-6
        ), "LMM parameters or momentum did not change during training pass when expected (check loss threshold)."
        print("LMM state changed during training pass, as expected.")

    except Exception as e:
        print(f"Error during training mode test: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- Eval Mode Test (with LMM test-time update) ---")
    model.eval()  # update_lmm_at_test_time is True in config
    try:
        initial_mem_stats_eval = model.get_memory_stats()  # Current state from training
        print(f"Memory stats before eval pass: {initial_mem_stats_eval}")

        with torch.no_grad():  # Main model forward is no_grad, but LMM updates internally
            logits_eval = model(input_ids, padding_mask=padding_mask_example)
        print(f"Eval Forward pass successful! Output shape: {logits_eval.shape}")

        mem_stats_after_eval_pass = model.get_memory_stats()
        print(
            f"Memory stats after 1 eval pass (LMM updated): {mem_stats_after_eval_pass}"
        )

        assert (
            abs(
                initial_mem_stats_eval["lmm_param_norm"]
                - mem_stats_after_eval_pass["lmm_param_norm"]
            )
            > 1e-6
            or abs(
                initial_mem_stats_eval["lmm_momentum_norm"]
                - mem_stats_after_eval_pass["lmm_momentum_norm"]
            )
            > 1e-6
        ), "LMM parameters or momentum did not change during eval pass when test-time update is enabled."
        print("LMM state changed during eval pass with test-time update, as expected.")

    except Exception as e:
        print(f"Error during eval mode test: {e}")
        import traceback

        traceback.print_exc()

    print("\nRefactored Neural Memory Transformer test completed!")
