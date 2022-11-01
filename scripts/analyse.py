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
import time
from hydra_zen import instantiate, make_config, to_yaml, load_from_yaml
from pathlib import Path
import os
import jax.numpy as np
from jax import random

from grassgp.utils import load_and_convert_to_samples_dict as load_data


# %%
def print_file(x: Path):
    with x.open("r") as f:
        print(f.read())


# %%
# Job Parameters
base_path = Path(os.getcwd())
output_folder = "multirun"
date = "2022-10-27"
time = "10-08-59"
id_num = "0"

# %%
job_path = base_path / output_folder / date / time / id_num
assert job_path.exists()

# load config in
config = load_from_yaml(job_path / ".hydra" / "config.yaml")

# get data and training data
dataset_path = Path(config.dataset_path) / "dataset.npz"
training_dataset_path = job_path / "training_data.npz"
assert dataset_path.exists()
assert training_dataset_path.exists()
data = load_data(dataset_path)
training_data = load_data(training_dataset_path)

# %%
# process data and training data
X_fine = np.array(data['X'])
s_fine = np.array(data['s'])
Ps_fine = np.array(data['Ps'])
Ys_fine = np.array(data['Ys'])
anchor_point = np.array(data['anchor_point'])

X = np.array(training_data['X'])
s = np.array(training_data['s'])
Ps = np.array(training_data['Ps'])
Ys = np.array(training_data['Ys'])

# %%
# get inference data
inference_path = job_path / "inference_data.npz"
assert inference_path.exists()
inference_data = load_data(inference_path)

# %%
inference_data.keys()

# %%
