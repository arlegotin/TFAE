import tensorflow as tf
from typing import Optional


class BaseBottleneck(tf.keras.layers.Layer):
    def __init__(
        self,
        latent_dim: int,
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.kernel_regularizer = kernel_regularizer
