import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                d_model,
                d_ff
            ),

            nn.GELU(),

            nn.Linear(
                d_ff,
                d_model
            )
        )

    def forward(self, x):

        return self.network(x)