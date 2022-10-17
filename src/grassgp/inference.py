import os
import time
# import jax.numpy as np
from numpyro.infer import MCMC, NUTS


# helper function for doing hmc inference
def run_inference(rng_key, mcmc_config, model, *args):
    num_warmup = mcmc_config['num_warmup']
    num_samples = mcmc_config['num_samples']
    num_chains = mcmc_config['num_chains']
    thinning = mcmc_config['thinning']
    init_strategy = mcmc_config['init_strategy']

    start = time.time()

    kernel = NUTS(model, init_strategy=init_strategy)

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, thinning=thinning,
                progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True)

    mcmc.run(rng_key, *args)

    mcmc.print_summary()
    print('\nMCMC elapsed time:', time.time() - start)
    # return mcmc.get_samples()
    return mcmc


# def save_jax_array_dict(path: str, samples: dict):
#     """function to save jax.numpy arrays in a dictionary

#     Parameters
#     ----------
#     path : str
#         path to where to save the arrays
#     samples : dict
#         dictionary containing jax.numpy arrays
#     """
#     code = "np.savez(path"
#     for key, value in samples.items():
#         assert isinstance(value, type(np.zeros((1,))))
#         array_name = key.replace('.', '_')
#         code += f", {array_name} = samples['{key}']"
    
#     code += ")"
    
#     # print(code)
#     exec(code)

# def safe_save_jax_array_dict(path: str, jax_dict: dict):
#     assert all([isinstance(value, type(np.zeros((1,)))) for value in jax_dict.values()])
#     
#     if path[-4:] != '.npz':
#         path += '.npz'
#     
#     if os.path.exists(path):
#         raise FileExistsError(f"File exists, saving would overwrite; please use another path.")
#     else:
#         np.savez(path, **jax_dict)
#
#
# def load_and_convert_to_samples_dict(path: str):
#     """function to load a .npz file and convert back to a samples dictionary
#
#     Parameters
#     ----------
#     path : str
#         path to .npz file
#
#     Returns
#     -------
#     dict
#         samples dictionary
#     """
#     samples_data = np.load(path)
#     
#     samples = {}
#     
#     for name in samples_data.files:
#         samples[name] = samples_data[name]
#     
#     return samples
