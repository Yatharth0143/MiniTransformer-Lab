import torch.nn as nn

from src.multi_head_attention import MultiHeadAttention
from src.feed_forward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1
    ):

        super().__init__()

        self.attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.feed_forward = FeedForward(
            d_model,
            d_ff
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        attention_output, attention_weights = \
            self.attention(
                x,
                x,
                x,
                mask
            )

        x = self.norm1(
            x + self.dropout(attention_output)
        )

        ff_output = self.feed_forward(x)

        x = self.norm2(
            x + self.dropout(ff_output)
        )

        return x, attention_weights