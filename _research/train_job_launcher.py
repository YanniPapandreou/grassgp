from hydra_zen import launch, to_yaml, load_from_yaml
from pathlib import Path
from train import Config, train
import jax.numpy as np


def print_file(x: Path):
    with x.open("r") as f:
        print(f.read())


(jobs,) = launch(
    Config,
    train,
    overrides=[
        "model.grass_config.length=0.1,0.25,0.5,1.0,5.0,10.0"
    ],
    multirun=True,
    version_base="1.1"
)


