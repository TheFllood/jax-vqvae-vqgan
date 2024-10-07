import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import functools
from jax import random
import os
import requests
from tqdm import tqdm
import hashlib

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path

class LPIPS(nn.Module):
    # Learned perceptual metric
    use_dropout: bool = True

    def setup(self):
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # VGG16 features
        self.net = VGG16(requires_grad=False)
        self.lins = [NetLinLayer(self.chns[i], use_dropout=self.use_dropout) for i in range(5)]
        
        
    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt_path = get_ckpt_path(name, "taming/modules/autoencoder/lpips")
        ckpt = checkpoints.restore_checkpoint(ckpt_path, target=None)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))
        return ckpt
    
    def __call__(self, input, target):
        in0_input = self.scaling_layer(input)
        in1_input = self.scaling_layer(target)
        outs0 = self.net(in0_input)
        outs1 = self.net(in1_input)
        
        res = []
        for kk in range(len(self.chns)):
            feats0 = normalize_tensor(outs0[kk])
            feats1 = normalize_tensor(outs1[kk])
            diffs = (feats0 - feats1) ** 2
            res.append(spatial_average(self.lins[kk](diffs), keepdim=True))
        
        val = functools.reduce(jnp.add, res)
        return val


class ScalingLayer(nn.Module):
    def setup(self):
        self.shift = jnp.array([-.030, -.088, -.188])[None, :, None, None]
        self.scale = jnp.array([.458, .448, .450])[None, :, None, None]

    def __call__(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    chn_in: int
    chn_out: int = 1
    use_dropout: bool = False

    def setup(self):
        self.conv = nn.Conv(self.chn_out, (1, 1), use_bias=False)
        if self.use_dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = lambda x, deterministic: x

    def __call__(self, x, deterministic=True):
        x = self.dropout(x, deterministic=deterministic)
        x = self.conv(x)
        return x


class VGG16(nn.Module):
    requires_grad: bool = False

    def setup(self):
        # Simulating VGG16 slices (you would need to load actual weights)
        self.slices = [
            nn.Sequential([nn.Conv(64, (3, 3)), nn.relu]),
            nn.Sequential([nn.Conv(128, (3, 3)), nn.relu]),
            nn.Sequential([nn.Conv(256, (3, 3)), nn.relu]),
            nn.Sequential([nn.Conv(512, (3, 3)), nn.relu]),
            nn.Sequential([nn.Conv(512, (3, 3)), nn.relu]),
        ]

    def __call__(self, x):
        outputs = []
        for slice in self.slices:
            x = slice(x)
            outputs.append(x)
        return outputs


def normalize_tensor(x, eps=1e-10):
    norm_factor = jnp.sqrt(jnp.sum(x**2, axis=1, keepdims=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return jnp.mean(x, axis=(2, 3), keepdims=keepdim)
