import tensorflow as tf
from tfae.utils import split_sampled_tensor


class BaseRegularizer(tf.keras.regularizers.Regularizer):
    """
    Base class for different VAE-regularizers.
    Inheriting classes should implement method "calc".
    """

    def __init__(self, beta: float = 1.0, parameters: int = 1):
        self.beta = beta
        self.parameters = parameters

    def __call__(self, x):
        return self.beta * self.calc(*split_sampled_tensor(x, self.parameters))

    def calc(self):
        raise NotImplementedError(
            'method "calc" of VariationalRegularizer has not been implemented'
        )
