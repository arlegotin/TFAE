import tensorflow as tf
from tfae.bottlenecks.base import BaseBottleneck
from .autoencoder import Autoencoder
from typing import Optional, Callable


class DeepAutoencoder(Autoencoder):
    def __init__(self, add_hidden: Callable, **kwargs):
        self.add_hidden = add_hidden
        self.hidden_num = 0

        super().__init__(**kwargs)

    def make_encoder(self, encoder_input):
        while True:
            self.hidden_num += 1

            encoder_input, finish = self.add_hidden(
                encoder_input,
                self.hidden_num,
                self.encoder_input_shape,
                self.latent_dim,
            )

            if finish:
                break

        return self.bottleneck(encoder_input)

    def make_decoder(self, decoder_input):
        for current_n in range(self.hidden_num, 0, -1):

            decoder_input, _ = self.add_hidden(
                decoder_input, current_n, self.encoder_input_shape, self.latent_dim
            )

        return tf.keras.layers.Dense(self.encoder_input_shape[-1])(decoder_input)
