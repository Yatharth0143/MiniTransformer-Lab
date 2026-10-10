
import torch

from src.tokenizer import CharTokenizer
from src.model import MiniTransformer


# A tiny sample corpus.
text = """
transformers learn patterns from sequences.
attention helps models understand context.
language models predict the next token.
"""

tokenizer = CharTokenizer(text)

config = {
    "vocab_size": tokenizer.vocab_size,
    "d_model": 64,
    "num_heads": 4,
    "num_layers": 2,
    "block_size": 32,
    "dropout": 0.1,
}

model = MiniTransformer(**config)

# Build one example of next-token prediction.
encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

x = encoded[:32].unsqueeze(0)
targets = encoded[1:33].unsqueeze(0)

logits, loss = model(x, targets)

print("Vocabulary size:", tokenizer.vocab_size)
print("Input shape:", tuple(x.shape))
print("Logits shape:", tuple(logits.shape))
print("Initial loss:", round(loss.item(), 4))

# Verify gradients work.
loss.backward()
print("Backpropagation: successful")

# Generate a short sample.
context = torch.tensor(
    [tokenizer.encode("transform")],
    dtype=torch.long,
)

generated = model.generate(context, max_new_tokens=50)
print("\nGenerated text:")
print(tokenizer.decode(generated[0].tolist()))