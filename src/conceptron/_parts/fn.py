import jax
import jax.numpy as jnp


def rotate_half(x: jax.Array) -> jax.Array:
    x1, x2 =  jnp.split(x, 2, axis=-1)
    return jnp.concat([-x2, x1], axis=-1)


__all__ = ['rotate_half']