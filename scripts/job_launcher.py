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
from pathlib import Path
def print_file(x: Path):
    with x.open("r") as f:
        print(f.read())


# %%
import os

# %%
path = Path(os.getcwd()) / "multirun" / "2022-10-21" / "13-37-25"

# %%
sorted(path.glob("*"))

# %%

# %%
for i in range(8):
    my_path = path / f"{i}"
    paths_list = sorted(my_path.glob("*"))
    if len(paths_list) == 2:
        print(i)
        print_file(my_path / ".hydra" / "overrides.yaml")

# %%
path 

# %%
job_dir = Path(job.working_dir)

# %%
job_dir

# %%
job_dir

# %%
sorted((job_dir / ".hydra").glob("*"))

# %%
print_file(job_dir / ".hydra" / "config.yaml")

# %%
sorted(job_dir.glob("*"))

# %%
from grassgp.plot_utils import plot_projected_data

# %%
from grassgp.utils import load_and_convert_to_samples_dict as load_data

# %%
data = load_data(str(job_dir / "dataset.npz"))

# %%
X = np.array(data['X'])
s = np.array(data['s'])
Ys = np.array(data['Ys'])
Ps = np.array(data['Ps'])
X_projs = np.einsum('ij,ljk->lik', X, Ps)

# %%
plot_projected_data(X_projs, s, Ys)

# %%
job_dir / ".hydra" / "config.yaml"

# %%
data_config = load_from_yaml(job_dir / ".hydra" / "config.yaml")

# %%
print(to_yaml(data_config))

# %%
n_s = s.shape[0]

# %%
D = X.shape[1]

# %%
Ps.shape

# %%
for i in range(D):
    plt.plot(s, Ps[:,i,0])
    plt.grid()
    # plt.ylim((-1,1))
    plt.show()

# %%
