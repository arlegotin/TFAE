import tensorflow as tf
from tfae.bottlenecks.base import BaseBottleneck


class VariationalBottleneck(BaseBottleneck):
    def __init__(self, parameters: int, sampler: tf.keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        self.sampled = tf.keras.layers.Dense(
            self.latent_dim * parameters,
            kernel_regularizer=self.kernel_regularizer,
        )

        self.sampler = sampler

    def call(self, x, training=False):

        x = self.sampled(x, training=training)

        x = self.sampler(x, training=training)

        return x
