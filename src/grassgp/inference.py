import os
import time
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
    return mcmc
