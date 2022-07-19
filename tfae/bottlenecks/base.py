import tensorflow as tf


class BaseBottleneck(tf.keras.layers.Layer):
    def __init__(
        self,
        latent_dim: int,
        kernel_regularizer: tf.keras.regularizers.Regularizer,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.kernel_regularizer = kernel_regularizer
