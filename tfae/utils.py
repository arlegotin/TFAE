import tensorflow as tf


def split_sampled_tensor(x: tf.Tensor, parameters: int):
    return tf.split(x, num_or_size_splits=parameters, axis=1)
