import numpy as np

from src.attention import scaled_dot_product_attention

#here we are going to test the scaled dot-product attention mechanism with some sample data.
#Q is query matrix what info we want to retrieve, K is key matrix (what info does it contain) , and V is value matrix [ what info should be provided].
Q = np.array([
    [1, 0, 1],
    [0, 1, 0]
], dtype=float)

K = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0]
], dtype=float)

V = np.array([
    [10, 0],
    [0, 10],
    [5, 5]
], dtype=float)


output, weights = scaled_dot_product_attention(Q, K, V)

print("Attention Weights:")
print(weights)

print("\nAttention Output:")
print(output)