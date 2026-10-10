
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        unknown = set(text) - set(self.stoi)
        if unknown:
            raise ValueError(f"Unknown characters: {unknown}")
        return [self.stoi[ch] for ch in text]

    def decode(self, token_ids):
        return "".join(self.itos[int(i)] for i in token_ids)