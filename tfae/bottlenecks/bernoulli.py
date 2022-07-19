import tensorflow as tf
from tfae.bottlenecks.variational import VariationalBottleneck


class BernoulliSampler(tf.keras.layers.Layer):
    def call(self, x, training=False):

        if not training:
            return x

        # Workaround for unknown batch size. Source:
        # https://datascience.stackexchange.com/questions/51086/valueerror-cannot-convert-a-partially-known-tensorshape-to-a-tensor-256
        eps = tf.keras.backend.random_uniform(shape=tf.shape(x))

        z = tf.math.less(eps, x)
        z = tf.cast(z, dtype=tf.float32)

        return z


class BernoulliBottleneck(VariationalBottleneck):
    def __init__(self, **kwargs):
        super().__init__(parameters=1, sampler=BernoulliSampler(), **kwargs)
