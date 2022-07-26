import tensorflow as tf
from tfae.bottlenecks.base import BaseBottleneck


class Autoencoder(tf.keras.Model):
    def __init__(self, bottleneck: BaseBottleneck, **kwargs):
        super().__init__(**kwargs)
        self.bottleneck = bottleneck

    def build(self, input_shape):
        self.encoder_input_shape = input_shape

        self.encoder_input = tf.keras.layers.Input(shape=input_shape[1:])
        self.decoder_input = tf.keras.layers.Input(shape=(self.latent_dim,))

        encoder = self.make_encoder(self.encoder_input)
        decoder = self.make_decoder(self.decoder_input)

        self.encoder = tf.keras.Model(inputs=self.encoder_input, outputs=encoder)
        self.decoder = tf.keras.Model(inputs=self.decoder_input, outputs=decoder)

    @property
    def latent_dim(self):
        return self.bottleneck.latent_dim

    def make_encoder(self, encoder_input):
        return self.bottleneck(encoder_input)

    def make_decoder(self, decoder_input):
        return tf.keras.layers.Dense(self.encoder_input_shape[-1])(decoder_input)

    def call(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
