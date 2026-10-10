
from pathlib import Path

import torch

from src.tokenizer import CharTokenizer


def load_dataset(
    file_path="data/tiny_corpus.txt",
    train_fraction=0.9,
):
    """Load text and split token IDs into training and validation data."""

    text = Path(file_path).read_text(encoding="utf-8")

    if len(text) < 20:
        raise ValueError("Dataset is too small. Add more training text.")

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    # Build the character vocabulary.
    tokenizer = CharTokenizer(text)
    token_ids = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long,
    )

    split_index = int(train_fraction * len(token_ids))

    train_data = token_ids[:split_index]
    val_data = token_ids[split_index:]

    return tokenizer, train_data, val_data


def get_batch(data, batch_size=4, block_size=32):
    """
    Sample input sequences x and next-token targets y.

    Each y sequence is x shifted one character to the right.
    """

    if len(data) <= block_size:
        raise ValueError(
            f"Need more than {block_size} tokens; "
            f"received {len(data)}."
        )

    # Random valid starting positions.
    starts = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,),
    )

    x = torch.stack([
        data[start : start + block_size]
        for start in starts.tolist()
    ])

    y = torch.stack([
        data[start + 1 : start + block_size + 1]
        for start in starts.tolist()
    ])

    return x, y