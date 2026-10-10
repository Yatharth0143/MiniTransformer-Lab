
import torch

from src.data_loader import load_dataset, get_batch


torch.manual_seed(42)

tokenizer, train_data, val_data = load_dataset()

BATCH_SIZE = 4
BLOCK_SIZE = 32

x, y = get_batch(
    train_data,
    batch_size=BATCH_SIZE,
    block_size=BLOCK_SIZE,
)

print("Vocabulary size:", tokenizer.vocab_size)
print("Total tokens:", len(train_data) + len(val_data))
print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))

print("\nInput batch shape:", tuple(x.shape))
print("Target batch shape:", tuple(y.shape))

print("\nFirst input sequence:")
print(tokenizer.decode(x[0].tolist()))

print("\nFirst target sequence:")
print(tokenizer.decode(y[0].tolist()))

# Verify that targets are shifted inputs.
assert x.shape == (BATCH_SIZE, BLOCK_SIZE)
assert y.shape == (BATCH_SIZE, BLOCK_SIZE)
assert torch.equal(x[:, 1:], y[:, :-1])

print("\nAll data pipeline checks passed!")