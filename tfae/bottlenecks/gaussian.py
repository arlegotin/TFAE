import tensorflow as tf
from tfae.bottlenecks.variational import VariationalBottleneck
from tfae.utils import split_sampled_tensor


class GaussianSampler(tf.keras.layers.Layer):
    def call(self, x, training=False):

        mean, logvar = split_sampled_tensor(x, 2)

        if not training:
            return mean

        # Workaround for unknown batch size. Source:
        # https://datascience.stackexchange.com/questions/51086/valueerror-cannot-convert-a-partially-known-tensorshape-to-a-tensor-256
        eps = tf.keras.backend.random_normal(shape=tf.shape(logvar))

        z = eps * tf.exp(logvar * 0.5) + mean

        return z


class GaussianBottleneck(VariationalBottleneck):
    def __init__(self, **kwargs):
        super().__init__(parameters=2, sampler=GaussianSampler(), **kwargs)
