import jax.numpy as np

def zero_mean(x):
  return 0.0

def reg_test_mean(x):
    y = x + 0.2 * (x ** 3) + 0.5 * ((0.5 + x) ** 2) * np.sin(4.0 * x)
    return y
