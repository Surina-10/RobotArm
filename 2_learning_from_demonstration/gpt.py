# ============================================================================
# APPROACH 2: the model, a small GPT-style transformer for robot motion
#
# Same idea as a language model predicting the next word, but here each "token"
# is a vector of 5 servo positions, and the model predicts the next 5 positions.
# Because positions are continuous numbers (not words), a Linear layer projects
# them in (instead of a word-embedding table) and the loss is mean squared
# error (instead of cross-entropy).
#
# Architecture: input projection + learned position embedding ->
# n_layer x [LayerNorm -> causal self-attention -> LayerNorm -> MLP] (with
# residual connections) -> LayerNorm -> Linear back to 5 servo values.
#
# Based on Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled
# out" (https://www.youtube.com/watch?v=kCc8FmEb1nY), which builds a character-
# level text GPT. I adapted it for robot motion: continuous servo vectors
# instead of characters, a Linear input projection instead of an embedding
# table, and MSE loss instead of cross-entropy.
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# Self-attention: each time step looks back at earlier steps to decide what comes next.
# The causal mask stops it from looking at the future.
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Causal mask so tokens only attend to past tokens
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


# One transformer layer: attention + a small feed-forward network, each with a residual connection.
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# The full model. input_dim=5 servos; d_model=hidden size; n_head=attention heads;
# n_layer=number of blocks; max_seq_len=longest motion it can handle.
class RobotGPT(nn.Module):
    def __init__(self, input_dim=5, d_model=128, n_head=4, n_layer=4, max_seq_len=500):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, input_dim)

    def forward(self, x, targets=None):
        B, T, _ = x.size()
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)

        # Input vector projection + positional embedding
        x = self.input_proj(x) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.head(x)  # Predicts next 5 servo values

        loss = None
        if targets is not None:
            loss = F.mse_loss(logits, targets)

        return logits, loss

