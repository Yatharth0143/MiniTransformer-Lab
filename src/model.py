
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Prevent each position from seeing future tokens.
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size))
            .view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # (batch, sequence, model_dim)
        # -> (batch, heads, sequence, head_dim)
        q = q.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        output = weights @ v

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, d_model)

        return self.proj(output)


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, block_size, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model, num_heads, block_size, dropout
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x):
        # Pre-layer normalization with residual connections.
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        block_size=128,
        dropout=0.1,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model, num_heads, block_size, dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, targets=None):
        batch_size, seq_len = token_ids.shape

        if seq_len > self.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"block size {self.block_size}"
            )

        positions = torch.arange(
            seq_len, device=token_ids.device
        )

        x = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

        x = self.blocks(x)
        x = self.final_norm(x)

        logits = self.lm_head(x)
        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens=100):
        self.eval()

        for _ in range(max_new_tokens):
            context = token_ids[:, -self.block_size:]
            logits, _ = self(context)

            # Predict the next token from the last position.
            next_token_logits = logits[:, -1, :]
            probabilities = F.softmax(next_token_logits, dim=-1)

            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)

        return token_ids