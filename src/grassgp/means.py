from functools import partial
import jax.numpy as np
from jax import jit

import chex


@partial(jit, static_argnums=(1, 2))
def zero_mean(s: chex.ArrayDevice, d: int, n: int) -> chex.ArrayDevice:
    return np.zeros((d, n))
