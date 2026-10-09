import numpy as np

from src.attention import scaled_dot_product_attention
#this is self attention mechanism 
#we have done 

np.random.seed(42)

X = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0]
])


d_model = X.shape[1]

W_Q = np.random.randn(d_model, d_model)
W_K = np.random.randn(d_model, d_model)
W_V = np.random.randn(d_model, d_model)


Q = X @ W_Q
K = X @ W_K
V = X @ W_V


output, weights = scaled_dot_product_attention(Q, K, V)


print("Q:")
print(Q)

print("\nK:")
print(K)

print("\nV:")
print(V)

print("\nAttention Weights:")
print(weights)

print("\nOutput:")
print(output)