from src.positional_encoding import positional_encoding

pe = positional_encoding(
    seq_len=10,
    d_model=8
)

print(pe)