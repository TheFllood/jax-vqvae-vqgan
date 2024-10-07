import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import numpy as np

class Normalize(nn.Module):
    in_channels: int
    num_groups: int = 32

    @nn.compact
    def __call__(self, x):
        return nn.GroupNorm(num_groups=self.num_groups, epsilon=1e-6)(x)

class Upsample(nn.Module):
    in_channels: int
    with_conv: bool = True

    def setup(self):
        if self.with_conv:
            self.conv = nn.Conv(features=self.in_channels, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))

    def __call__(self, x):
        x = jax.image.resize(x, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), method='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    in_channels: int
    with_conv: bool = True

    def setup(self):
        if self.with_conv:
            self.conv = nn.Conv(features=self.in_channels, kernel_size=(3, 3), strides=(2, 2), padding='SAME')

    def __call__(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        return x

class ResnetBlock(nn.Module):
    in_channels: int
    out_channels: int = None
    conv_shortcut: bool = False
    dropout: float = 0.0
    temb_channels: int = 512

    def setup(self):
        self.out_channels = self.in_channels if self.out_channels is None else self.out_channels
        self.use_conv_shortcut = self.conv_shortcut

        self.norm1 = Normalize(self.in_channels)
        self.conv1 = nn.Conv(features=self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        if self.temb_channels > 0:
            self.temb_proj = nn.Dense(features=self.out_channels)
        self.norm2 = Normalize(self.out_channels)
        self.dropout_layer = nn.Dropout(rate=self.dropout)
        self.conv2 = nn.Conv(features=self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv(features=self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
            else:
                self.nin_shortcut = nn.Conv(features=self.out_channels, kernel_size=(1, 1), strides=(1, 1), padding='VALID')

    def __call__(self, x, temb):
        h = self.norm1(x)
        h = nn.relu(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nn.relu(temb))[:, None, None, :]

        h = self.norm2(h)
        h = nn.relu(h)
        h = self.dropout_layer(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class LinAttnBlock(nn.Module):
    in_channels: int

    def setup(self):
        self.dim = self.in_channels
        self.heads = 1
        self.dim_head = self.in_channels
        # Define other necessary parameters and setup for linear attention.

    def __call__(self, x):
        # Implement the forward pass logic for linear attention here.
        pass

class AttnBlock(nn.Module):
    in_channels: int

    def setup(self):
        self.norm = Normalize(self.in_channels)
        self.q = nn.Conv(features=self.in_channels, kernel_size=(1, 1))
        self.k = nn.Conv(features=self.in_channels, kernel_size=(1, 1))
        self.v = nn.Conv(features=self.in_channels, kernel_size=(1, 1))
        self.proj_out = nn.Conv(features=self.in_channels, kernel_size=(1, 1))

    def __call__(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).transpose((0, 2, 1))
        k = k.reshape(b, c, h * w)
        w_ = jnp.matmul(q, k) * (c ** -0.5)
        w_ = nn.softmax(w_, axis=-1)

        # attend to values
        v = v.reshape(b, c, h * w)
        h_ = jnp.matmul(v, w_.transpose((0, 2, 1)))
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        return x + h_

def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity()
    else:
        return LinAttnBlock(in_channels)
    

class Encoder(nn.Module):
    ch: int
    out_ch: int
    ch_mult: tuple = (1, 2, 4, 8)
    num_res_blocks: int
    attn_resolutions: list
    dropout: float = 0.0
    resamp_with_conv: bool = True
    in_channels: int
    resolution: int
    z_channels: int
    double_z: bool = True
    use_linear_attn: bool = False
    attn_type: str = "vanilla"

    def setup(self):
        if self.use_linear_attn:
            self.attn_type = "linear"
        self.num_resolutions = len(self.ch_mult)
        self.in_ch_mult = (1,) + tuple(self.ch_mult)
        self.curr_res = self.resolution
        
        # Downsampling layers
        self.conv_in = nn.Conv(features=self.ch, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        self.down = []
        for i_level in range(self.num_resolutions):
            block = []
            attn = []
            block_in = self.ch * self.in_ch_mult[i_level]
            block_out = self.ch * self.ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=self.dropout))
                block_in = block_out
                if self.curr_res in self.attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=self.attn_type))
            down = dict(block=block, attn=attn)
            if i_level != self.num_resolutions - 1:
                down['downsample'] = nn.Conv(features=block_in, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
                self.curr_res = self.curr_res // 2
            self.down.append(down)
        
        # Middle layers
        self.mid_block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=self.dropout)
        self.mid_attn_1 = make_attn(block_in, attn_type=self.attn_type)
        self.mid_block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=self.dropout)
        
        # End layers
        self.norm_out = nn.LayerNorm()
        self.conv_out = nn.Conv(features=2 * self.z_channels if self.double_z else self.z_channels, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))

    def __call__(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for block in self.down[i_level]['block']:
                h = block(h)
            for attn in self.down[i_level]['attn']:
                h = attn(h)
            if 'downsample' in self.down[i_level]:
                h = self.down[i_level]['downsample'](h)
        
        # Middle
        h = self.mid_block_1(h)
        h = self.mid_attn_1(h)
        h = self.mid_block_2(h)
        
        # End
        h = self.norm_out(h)
        h = nn.relu(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    ch: int
    out_ch: int
    ch_mult: tuple = (1, 2, 4, 8)
    num_res_blocks: int
    attn_resolutions: list
    dropout: float = 0.0
    resamp_with_conv: bool = True
    in_channels: int
    resolution: int
    z_channels: int
    give_pre_end: bool = False
    tanh_out: bool = False
    use_linear_attn: bool = False
    attn_type: str = "vanilla"

    def setup(self):
        if self.use_linear_attn:
            self.attn_type = "linear"
        self.num_resolutions = len(self.ch_mult)
        self.in_ch_mult = (1,) + tuple(self.ch_mult)
        self.curr_res = self.resolution // 2 ** (self.num_resolutions - 1)
        self.block_in = self.ch * self.ch_mult[self.num_resolutions - 1]
        
        # Z to block_in
        self.conv_in = nn.Conv(features=self.block_in, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        
        # Middle layers
        self.mid_block_1 = ResnetBlock(in_channels=self.block_in, out_channels=self.block_in, dropout=self.dropout)
        self.mid_attn_1 = make_attn(self.block_in, attn_type=self.attn_type)
        self.mid_block_2 = ResnetBlock(in_channels=self.block_in, out_channels=self.block_in, dropout=self.dropout)
        
        # Upsampling layers
        self.up = []
        for i_level in reversed(range(self.num_resolutions)):
            block = []
            attn = []
            block_out = self.ch * self.ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=self.block_in, out_channels=block_out, dropout=self.dropout))
                self.block_in = block_out
                if self.curr_res in self.attn_resolutions:
                    attn.append(make_attn(self.block_in, attn_type=self.attn_type))
            up = dict(block=block, attn=attn)
            if i_level != 0:
                up['upsample'] = nn.ConvTranspose(features=self.block_in, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
                self.curr_res = self.curr_res * 2
            self.up.insert(0, up)
        
        # End layers
        self.norm_out = nn.LayerNorm()
        self.conv_out = nn.Conv(features=self.out_ch, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))

    def __call__(self, z):
        h = self.conv_in(z)
        
        # Middle
        h = self.mid_block_1(h)
        h = self.mid_attn_1(h)
        h = self.mid_block_2(h)
        
        # Upsampling
        for i_level in range(self.num_resolutions):
            for block in self.up[i_level]['block']:
                h = block(h)
            for attn in self.up[i_level]['attn']:
                h = attn(h)
            if 'upsample' in self.up[i_level]:
                h = self.up[i_level]['upsample'](h)
        
        # End
        if self.give_pre_end:
            return h
        
        h = self.norm_out(h)
        h = nn.relu(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = jnp.tanh(h)
        return h


class DiagonalGaussianDistribution:
    def __init__(self, moments):
        self.mean, self.logvar = jnp.split(moments, 2, axis=-1)

    def sample(self, rng):
        std = jnp.exp(0.5 * self.logvar)
        return self.mean + std * jax.random.normal(rng, self.mean.shape)

    def mode(self):
        return self.mean

class AutoencoderKL(nn.Module):
    ddconfig: dict
    lossconfig: dict
    embed_dim: int
    colorize_nlabels: int = None
    monitor: str = None

    def setup(self):
        self.encoder = Encoder(**self.ddconfig)
        self.decoder = Decoder(**self.ddconfig)
        self.quant_conv = nn.Conv(features=2 * self.embed_dim, kernel_size=(1, 1))
        self.post_quant_conv = nn.Conv(features=self.ddconfig["z_channels"], kernel_size=(1, 1))
        if self.colorize_nlabels is not None:
            self.colorize = self.param('colorize', jax.random.normal, (3, self.colorize_nlabels, 1, 1))

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def __call__(self, input, sample_posterior=True, rng=None):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample(rng)
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.colorize = self.param('colorize', jax.random.normal, (3, x.shape[1], 1, 1))
        x = nn.Conv(features=3, kernel_size=(1, 1), use_bias=False, kernel_init=lambda *_: self.colorize)(x)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

# Training utilities
def create_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones([1, 256, 256, 3]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch, rng):
    def loss_fn(params):
        reconstructions, posterior = state.apply_fn({'params': params}, batch['image'], rng=rng)
        loss = jnp.mean((reconstructions - batch['image'])**2)  # Example loss (MSE)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss
