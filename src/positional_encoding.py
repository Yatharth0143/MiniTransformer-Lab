import numpy as np

#in this we add information about the position of each token in the sequence to the input embeddings.
def positional_encoding(seq_len, d_model):

    position = np.arange(seq_len)[:, np.newaxis]

    dimension = np.arange(d_model)[np.newaxis, :]

    angle_rates = 1 / np.power(
        10000,
        (2 * (dimension // 2)) / d_model
    )

    angles = position * angle_rates

    encoding = np.zeros((seq_len, d_model))

    encoding[:, 0::2] = np.sin(
        angles[:, 0::2]
    )

    encoding[:, 1::2] = np.cos(
        angles[:, 1::2]
    )

    return encoding