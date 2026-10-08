import numpy as np


def softmax(x):
    """
    in this step we are going to implement the softmax function, 
    which is used to convert the scores into probabilities.
     The softmax function takes a vector of real numbers as input and normalizes it into a probability distribution. 
     The output values will be in the range (0, 1) and will sum to 1.
    """
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

#these all are written by meeeee not ai for my understanding and learning purpose only.
def scaled_dot_product_attention(Q, K, V):
    """
    second step is to implement the scaled dot-product attention mechanism.
    between the query (Q), key (K), and value (V) matrices.
    """

    d_k = K.shape[-1]

    scores = Q @ K.T

    scaled_scores = scores / np.sqrt(d_k)

    attention_weights = softmax(scaled_scores)

    output = attention_weights @ V

    return output, attention_weights