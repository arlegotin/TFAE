import tensorflow as tf
from tfae.regularizers.base import BaseRegularizer


@tf.keras.utils.register_keras_serializable(package="TFAE", name="L2Regularizer")
class L2Regularizer(BaseRegularizer):
    def calc(self, x):
        return tf.math.reduce_mean(tf.math.square(x))
