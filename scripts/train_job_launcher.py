# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: grassgp
#     language: python
#     name: grassgp
# ---

# %%
from hydra_zen import launch, to_yaml, load_from_yaml
from train import Config, train
import jax.numpy as np

# %%
print(to_yaml(Config))

# %%
