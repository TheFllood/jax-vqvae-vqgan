import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from discriminator import NLayerDiscriminator
from model.lpips import LPIPS

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

class LPIPSWithDiscriminator(nn.Module):
    disc_start: int
    logvar_init: float = 0.0
    kl_weight: float = 1.0
    pixelloss_weight: float = 1.0
    disc_num_layers: int = 3
    disc_in_channels: int = 3
    disc_factor: float = 1.0
    disc_weight: float = 1.0
    perceptual_weight: float = 1.0
    use_actnorm: bool = False
    disc_conditional: bool = False
    disc_loss: str = "hinge"

    def setup(self):
        assert self.disc_loss in ["hinge", "vanilla"]
        self.perceptual_loss = LPIPS().eval()
        self.logvar = self.param('logvar', lambda rng, shape: jnp.ones(shape) * self.logvar_init, ())
        self.discriminator = NLayerDiscriminator(input_nc=self.disc_in_channels,
                                                 n_layers=self.disc_num_layers,
                                                 use_actnorm=self.use_actnorm)
        self.disc_loss_fn = hinge_d_loss if self.disc_loss == "hinge" else vanilla_d_loss

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = jax.grad(lambda params: nll_loss)(last_layer)
            g_grads = jax.grad(lambda params: g_loss)(last_layer)
        else:
            nll_grads = jax.grad(lambda params: nll_loss)(self.last_layer[0])
            g_grads = jax.grad(lambda params: g_loss)(self.last_layer[0])

        d_weight = jnp.linalg.norm(nll_grads) / (jnp.linalg.norm(g_grads) + 1e-4)
        d_weight = jnp.clip(d_weight, 0.0, 1e4)
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def __call__(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = jnp.abs(inputs - reconstructions)
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs, reconstructions)
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / jnp.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = jnp.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = jnp.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = jnp.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions)
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(jnp.concatenate((reconstructions, cond), axis=1))
            g_loss = -jnp.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = jnp.array(0.0)
            else:
                d_weight = jnp.array(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.disc_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss,
                   "{}/logvar".format(split): self.logvar,
                   "{}/kl_loss".format(split): kl_loss,
                   "{}/nll_loss".format(split): nll_loss,
                   "{}/rec_loss".format(split): jnp.mean(rec_loss),
                   "{}/d_weight".format(split): d_weight,
                   "{}/disc_factor".format(split): disc_factor,
                   "{}/g_loss".format(split): g_loss}
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs)
                logits_fake = self.discriminator(reconstructions)
            else:
                logits_real = self.discriminator(jnp.concatenate((inputs, cond), axis=1))
                logits_fake = self.discriminator(jnp.concatenate((reconstructions, cond), axis=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.disc_start)
            d_loss = disc_factor * self.disc_loss_fn(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss,
                   "{}/logits_real".format(split): jnp.mean(logits_real),
                   "{}/logits_fake".format(split): jnp.mean(logits_fake)}
            return d_loss, log