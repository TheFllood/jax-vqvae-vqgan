import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any


def weights_init(module):
    """
    Initialize weights similarly to PyTorch version.
    """
    if isinstance(module, nn.Conv):
        module.kernel = jax.random.normal(jax.random.PRNGKey(0), module.kernel.shape) * 0.02
    elif isinstance(module, nn.BatchNorm):
        module.scale = jax.random.normal(jax.random.PRNGKey(0), module.scale.shape) * 0.02
        module.bias = jnp.zeros(module.bias.shape)


class ActNorm(nn.Module):
    num_features: int
    logdet: bool = False
    allow_reverse_init: bool = False

    @nn.compact
    def __call__(self, x, reverse: bool = False):
        scale = self.param('scale', nn.initializers.ones, (1, self.num_features, 1, 1))
        bias = self.param('bias', nn.initializers.zeros, (1, self.num_features, 1, 1))

        if reverse:
            return (x - bias) / (scale + 1e-6)
        else:
            return scale * x + bias


class AbstractEncoder(nn.Module):
    def encode(self, *args, **kwargs):
        raise NotImplementedError


class Labelator(AbstractEncoder):
    n_classes: int
    quantize_interface: bool = True

    @nn.compact
    def __call__(self, c):
        c = jnp.expand_dims(c, axis=-1)
        if self.quantize_interface:
            return c, None, [None, None, c.astype(jnp.int32)]
        return c


class SOSProvider(AbstractEncoder):
    sos_token: int
    quantize_interface: bool = True

    @nn.compact
    def __call__(self, x):
        c = jnp.ones((x.shape[0], 1)) * self.sos_token
        c = c.astype(jnp.int32)
        if self.quantize_interface:
            return c, None, [None, None, c]
        return c


class NLayerDiscriminator(nn.Module):
    input_nc: int = 3
    ndf: int = 64
    n_layers: int = 3
    use_actnorm: bool = False

    @nn.compact
    def __call__(self, x):
        norm_layer = ActNorm if self.use_actnorm else nn.BatchNorm

        kw = 4
        padw = 1

        # First layer
        x = nn.Conv(features=self.ndf, kernel_size=(kw, kw), strides=(2, 2), padding=((padw, padw), (padw, padw)))(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        # Hidden layers
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, self.n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            x = nn.Conv(features=self.ndf * nf_mult, kernel_size=(kw, kw), strides=(2, 2), padding=((padw, padw), (padw, padw)))(x)
            x = norm_layer(num_features=self.ndf * nf_mult)(x)
            x = nn.leaky_relu(x, negative_slope=0.2)

        # Final hidden layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** self.n_layers, 8)
        x = nn.Conv(features=self.ndf * nf_mult, kernel_size=(kw, kw), strides=(1, 1), padding=((padw, padw), (padw, padw)))(x)
        x = norm_layer(num_features=self.ndf * nf_mult)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        # Output layer
        x = nn.Conv(features=1, kernel_size=(kw, kw), strides=(1, 1), padding=((padw, padw), (padw, padw)))(x)
        return x
