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
# # %load_ext autoreload
# # %autoreload 2

# %%
# import os
# from pathlib import Path

from hydra_zen import launch 
from train import Config, train
import jax.numpy as np

# %%
dataset_paths = [f"{str(Config.dataset_path)[:-1]}{i}" for i in range(8)]

dataset_paths.append('/home/yanni/projects/grassgp/scripts/outputs/2022-11-02/16-27-53/')

dataset_paths_override = ""
for i, path in enumerate(dataset_paths):
    if i != 8:
        dataset_paths_override += f"{path},"
    else:
        dataset_paths_override += f"{path}"

assert dataset_paths_override.split(',') == dataset_paths

# %%
(jobs,) = launch(
    Config,
    train,
    overrides=[
        f"train.n_subsample_gap=2",
        f"dataset_path={dataset_paths_override}",
        # "subsample_conf.random=True,False",
        # f"inner_model.grass_config.anchor_point={np.eye(2,1).tolist()},{np.array([[0.],[1.]]).tolist()}",
        "inner_model.grass_config.reorthonormalize=True,False",
        f"inner_model.grass_config.Omega=null,{np.eye(2).tolist()}",
        f"inner_model.grass_config.proj_locs=null,{np.zeros(2*1*5).tolist()}"
    ],
    multirun=True,
    version_base="1.1"
)
