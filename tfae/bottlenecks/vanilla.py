import tensorflow as tf
from tfae.bottlenecks.base import BaseBottleneck


class VanillaBottleneck(BaseBottleneck):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.layer = tf.keras.layers.Dense(
            self.latent_dim,
            kernel_regularizer=self.kernel_regularizer,
        )

    def call(self, x, training=False):

        x = self.layer(x, training=training)

        return x
