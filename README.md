# grassgp
Code for dimension-reduction project.

# Installation

## Recommended method (nix flake):

(*Note this has been tested on Ubuntu 22.04 and NixOS only but should work as long as you have nix installed*)

Steps:

1. Ensure `nix` is installed following the instructions [here](https://nixos.org/download.html). (**Note**: If you are using NixOS you can skip this step.) 
2. Enable `nix` flakes following the instructions [here](https://nixos.wiki/wiki/Flakes).
3. Clone the repo
4. `cd` into the repo (`cd grassgp`)
5. Activate the nix flake by running `nix develop` (this might take a while).
6. Use [`poetry`](https://python-poetry.org/) to install the Python source code and the Python source dependencies: `poetry install`
7. (Optional): Install the Julia source code to generate the data for the localised active subspace examples:
    1. Activate a Julia REPL: `julia`
    2. Open the Julia `pkg` REPL by pressing `]` from the Julia REPL. 
    3. Activate the environment contained in the repo root: `activate .`
    4. Instantiate the enviroment via: `instantiate` (this might take a while).
8. You can now spawn a Python shell with all the dependencies using `poetry shell`.
9. (*Alternatively/recommended*): you can launch a Jupyterlab environment containing the Python and Julia dependencies using: `poetry run jupyter lab`

## Alternative method (Generic Linux - without nix):
Steps:

1. Clone the repo
2. `cd` into the repo (`cd grassgp`)
3. Create a virtual environment
4. Install [`poetry`](https://python-poetry.org/) -- suggested method:
    1. Install [`pipx`](https://pypa.github.io/pipx/)
    2. Install `poetry` with `pipx install poetry`
5. *(Optional/recommended)*: upgrade pip with `pip install --upgrade pip`
6. Activate virtual environment.
7. Use `poetry` to install: `poetry install`
8. (Optional): Install the Julia source code to generate the data for the localised active subspace examples:
    1. Install Julia stable following the instructions [here](https://julialang.org/downloads/).
    2. Activate a Julia REPL: `julia`
    3. Open the Julia `pkg` REPL by pressing `]` from the Julia REPL. 
    4. Activate the environment contained in the repo root: `activate .`
    5. Instantiate the enviroment via: `instantiate` (this might take a while).
9. You can now spawn a Python shell with all the dependencies using `poetry shell`.
10. (*Alternatively/recommended*): you can launch a Jupyterlab environment containing the Python and Julia dependencies using: `poetry run jupyter lab`
