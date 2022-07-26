import tensorflow as tf
from tfae.regularizers.base import BaseRegularizer


@tf.keras.utils.register_keras_serializable(
    package="TFAE", name="GaussianKLDRegularizer"
)
class GaussianKLDRegularizer(BaseRegularizer):
    """
    This class regularizes a layer using
    KL-divergence between normal distibutions N(mean, exp(logvar)) and N(0, 1)
    """

    def __init__(self, **kwargs):
        super().__init__(parameters=2, **kwargs)

    def calc(self, mean, logvar):
        return 0.5 * tf.math.reduce_mean(
            tf.math.square(mean) + tf.math.exp(logvar) - logvar - 1
        )


@tf.keras.utils.register_keras_serializable(
    package="TFAE", name="GaussianReversedKLDRegularizer"
)
class GaussianReversedKLDRegularizer(BaseRegularizer):
    """
    This class regularizes a layer using
    reversed KL-divergence between normal distibutions N(mean, exp(logvar)) and N(0, 1)
    """

    def calc(self, mean, logvar):
        return 0.5 * tf.math.reduce_mean(
            (1 + tf.math.square(mean)) / tf.math.exp(logvar) + logvar - 1
        )
