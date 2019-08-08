import tensorflow as tf

def create_MLP(in_size, out_size, hidden_sizes, activation):
    mlp = []

    for layer in range(len(hidden_sizes) + 1):
        if layer == len(hidden_sizes):
            mlp.append(tf.keras.layers.Dense(out_size))
        elif layer == 0:
            mlp.append(tf.keras.layers.Dense(hidden_sizes[0],
                activation=activation, input_shape=[in_size]))
        else:
            mlp.append(tf.keras.layers.Dense(hidden_sizes[layer],
                activation=activation))

    mlp = tf.keras.Sequential(mlp)

    return mlp