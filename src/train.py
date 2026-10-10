
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data_loader import get_batch, load_dataset
from src.model import MiniTransformer


def estimate_loss(model, train_data, val_data, batch_size, block_size,
                  eval_iters, device):
    """Estimate average loss on training and validation batches."""
    model.eval()
    results = {}

    with torch.no_grad():
        for split, data in [
            ("train", train_data),
            ("val", val_data),
        ]:
            losses = []

            for _ in range(eval_iters):
                x, y = get_batch(
                    data, batch_size=batch_size,
                    block_size=block_size,
                )
                x, y = x.to(device), y.to(device)

                _, loss = model(x, y)
                losses.append(loss.item())

            results[split] = sum(losses) / len(losses)

    model.train()
    return results


def train():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Training device:", device)

    tokenizer, train_data, val_data = load_dataset()

    # Keep these settings consistent with the model's context window.
    batch_size = 16
    block_size = 32
    max_steps = 500
    eval_interval = 50
    eval_iters = 10
    learning_rate = 3e-4

    if len(train_data) <= block_size or len(val_data) <= block_size:
        raise ValueError(
            "Training and validation splits must each contain "
            "more tokens than block_size. Use a larger corpus."
        )

    config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 2,
        "block_size": block_size,
        "dropout": 0.1,
    }

    model = MiniTransformer(**config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Training tokens:", len(train_data))
    print("Validation tokens:", len(val_data))

    for step in range(max_steps + 1):
        if step % eval_interval == 0 or step == max_steps:
            losses = estimate_loss(
                model, train_data, val_data,
                batch_size, block_size, eval_iters, device,
            )
            print(
                f"Step {step:4d} | "
                f"train loss: {losses['train']:.4f} | "
                f"validation loss: {losses['val']:.4f}"
            )

        if step == max_steps:
            break

        x, y = get_batch(
            train_data,
            batch_size=batch_size,
            block_size=block_size,
        )
        x, y = x.to(device), y.to(device)

        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Helps prevent unusually large gradient updates.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    # Save everything required to reload the model later.
    Path("checkpoints").mkdir(exist_ok=True)
    checkpoint_path = Path("checkpoints/mini_gpt.pt")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config,
            "tokenizer_chars": tokenizer.chars,
            "final_train_loss": losses["train"],
            "final_val_loss": losses["val"],
        },
        checkpoint_path,
    )

    print(f"\nTraining complete! Saved to {checkpoint_path}")


if __name__ == "__main__":
    train()