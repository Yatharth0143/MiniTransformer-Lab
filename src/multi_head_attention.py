import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):

        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads

        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(
            d_model,
            d_model
        )

        self.W_k = nn.Linear(
            d_model,
            d_model
        )

        self.W_v = nn.Linear(
            d_model,
            d_model
        )

        self.W_o = nn.Linear(
            d_model,
            d_model
        )

    def forward(self, Q, K, V, mask=None):

        batch_size = Q.size(0)

        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        Q = Q.view(
            batch_size,
            -1,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            -1,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            -1,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1)

        scores = scores / (
            self.head_dim ** 0.5
        )

        if mask is not None:
            scores = scores.masked_fill(
                mask == 0,
                float("-inf")
            )

        attention = torch.softmax(
            scores,
            dim=-1
        )

        output = attention @ V

        output = output.transpose(1, 2)

        output = output.contiguous().view(
            batch_size,
            -1,
            self.d_model
        )

        return self.W_o(output), attention