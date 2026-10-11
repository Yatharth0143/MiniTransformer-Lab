import torch
from src.model import MiniTransformer


def main():
    checkpoint_path = "checkpoints/mini_gpt.pt"

    # Load the saved model checkpoint.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["model_config"]

    # Rebuild the model architecture and restore its trained weights.
    model = MiniTransformer(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Restore the character vocabulary used during training.
    chars = checkpoint["tokenizer_chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    prompt = input("Enter a starting prompt: ")

    if not prompt:
        prompt = "Transformers"

    unknown = sorted(set(prompt) - set(stoi))
    if unknown:
        raise ValueError(
            f"These characters are not in the vocabulary: {unknown}"
        )

    input_ids = torch.tensor(
        [[stoi[ch] for ch in prompt]],
        dtype=torch.long,
    )

    # Generate new characters from the trained model.
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=100,
        )

    output = "".join(
        itos[int(token_id)]
        for token_id in generated_ids[0]
    )

    print("\n--- Generated Text ---")
    print(output)


if __name__ == "__main__":
    main()
