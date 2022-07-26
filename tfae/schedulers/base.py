import tensorflow as tf
from typing import Callable, Optional


class BaseScheduler(tf.keras.callbacks.Callback):
    """
    Represents base variable
    """

    def __init__(
        self,
        skew: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.skew = skew
        self._value = None

    @property
    def initial_value(self) -> float:
        raise NotImplementedError('property "initial_value" was not implemented')

    @property
    def duration(self) -> int:
        raise NotImplementedError('property "duration" was not implemented')

    def calc(self, n: int) -> float:
        raise NotImplementedError('method "calc" was not implemented')

    @property
    def value(self) -> tf.Variable:
        if self._value is None:
            self._value = tf.Variable(
                initial_value=self.initial_value, trainable=False, dtype=tf.float32
            )

        return self._value

    def update(self, n: int) -> None:
        if n > self.duration + 1:
            return

        next_value = self.calc(n)

        if self.skew is not None:
            next_value = self.skew(next_value)

        self.value.assign(next_value)

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None):
        self.update(epoch)
