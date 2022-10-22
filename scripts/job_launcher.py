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
from generate_dataset import Config, generate_dataset
import jax.numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,8)

# %%
print(to_yaml(Config))

# %%
(jobs,) = launch(
    Config,
    generate_dataset,
    overrides=[
        "inner_model.grass_config.reorthonormalize=True,False",
        f"inner_model.grass_config.Omega=null,{np.eye(2).tolist()}",
        f"inner_model.grass_config.proj_locs=null,{np.zeros(2*1*10).tolist()}"
    ],
    multirun=True,
    version_base="1.1"
)

# %%
job_rotate = launch(
    Config,
    generate_dataset,
    overrides=[
        "gen_projs_from_prior=False"
    ],
    version_base="1.1"
)

# %%
