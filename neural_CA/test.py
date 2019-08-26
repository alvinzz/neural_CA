import tensorflow as tf

optimizer = tf.keras.optimizers.Adam()
p = tf.Variable([[0,1],[1,0]], dtype=tf.float32)

@tf.function
def main():
    a = tf.constant([[0]], dtype=tf.float32)
    b = tf.constant([[1]], dtype=tf.float32)
    c = tf.concat([a, b], 0)
    c = tf.matmul(p, c)
    inner_loss = tf.reduce_sum(c)
    a_grad = tf.gradients(inner_loss, a)[0]
    updated_a = a - a_grad

    updated_c = tf.concat([updated_a, b], 0)
    loss = tf.reduce_sum(updated_c)

    grads = tf.gradients(loss, p)

    optimizer.apply_gradients(zip(grads, [p]))

    #print(tf.Graph.UPDATE_OPS)

main()

print(p)
