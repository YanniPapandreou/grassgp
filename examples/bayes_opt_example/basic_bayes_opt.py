# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Enable Float64 for more stable matrix inversions.
from jax.config import config

config.update("jax_enable_x64", True)

from jax import jit
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax as ox
from utils import clean_legend

from jaxopt import ScipyBoundedMinimize

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
    
from tqdm import tqdm

key = jr.PRNGKey(123)
plt.style.use("./gpjax.mplstyle")
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]


# %%
def fit_gp(x,y,key,kernel=gpx.kernels.Matern52(),meanf=gpx.mean_functions.Zero(),obs_noise=0.0,plot_training_loss=False):
    D = gpx.Dataset(X=x, y=y)
    prior = gpx.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.Gaussian(num_datapoints=D.n, obs_noise=obs_noise)
    posterior = prior * likelihood
    
    negative_mll = jit(gpx.objectives.ConjugateMLL(negative=True),static_argnums=1)
    negative_mll = gpx.objectives.ConjugateMLL(negative=True)
    negative_mll(posterior, train_data=D)
    
    opt_posterior, train_history = gpx.fit(
        model=posterior,
        objective=negative_mll,
        train_data=D,
        optim=ox.adam(learning_rate=0.01),
        num_iters=500,
        safe=True,
        verbose=False,
        key=key,
    )
    
    if plot_training_loss:
        fig, ax = plt.subplots()
        ax.plot(train_history, color=cols[1])
        ax.set(xlabel="Training iteration", ylabel="Negative marginal log likelihood")
        plt.show()
    
    return opt_posterior


# %%
def plot_posterior_and_acquisition(xtest,opt_posterior,xtrain,ytrain,beta=1.96):
    D = gpx.Dataset(X=xtrain, y=ytrain)
    
    latent_dist = opt_posterior.predict(xtest, train_data=D)
    predictive_dist = opt_posterior.likelihood(latent_dist)

    predictive_mean = predictive_dist.mean()
    predictive_std = predictive_dist.stddev()
    
    def lcb(x,posterior,data,beta=beta):
        x = x.reshape(-1,1)
        latent_dist = posterior.predict(x, train_data=data)
        predictive_dist = posterior.likelihood(latent_dist)

        mu = predictive_dist.mean()
        sigma = predictive_dist.stddev()

        return mu - beta * sigma
    
    fig, axs = plt.subplots(2,1,figsize=(7.5, 6))
    axs[0].plot(xtrain, ytrain, "x", label="Observations", color=cols[0], alpha=0.5)
    axs[0].fill_between(
        xtest.squeeze(),
        predictive_mean - beta * predictive_std,
        predictive_mean + beta * predictive_std,
        alpha=0.2,
        label=f"{beta:.2f} sigma",
        color=cols[1],
    )
    axs[0].plot(
        xtest,
        predictive_mean - beta * predictive_std,
        linestyle="--",
        linewidth=1,
        color=cols[1],
    )
    axs[0].plot(
        xtest,
        predictive_mean + beta * predictive_std,
        linestyle="--",
        linewidth=1,
        color=cols[1],
    )
    axs[0].plot(
        xtest, ytest, label="Latent function", color=cols[0], linestyle="--", linewidth=2
    )
    axs[0].plot(xtest, predictive_mean, label="Predictive mean", color=cols[1])
    axs[0].legend(loc="center left", bbox_to_anchor=(0.975, 0.5))

    axs[1].plot(x, y, "x", label="Observations", color=cols[0], alpha=0.5)
    axs[1].plot(xtest, lcb(xtest, opt_posterior, D), color=cols[1],label="LCB Acquisition function")
    axs[1].legend()
    axs[1].set_title("LCB Acquistion function")

    plt.show()


# %%
def optimise_acquisition(x0_s,opt_posterior,xtrain,ytrain,lower,upper,beta=1.96):
    d_in = xtrain.shape[1]
    assert x0_s.shape[1] == d_in
    D = gpx.Dataset(X=xtrain, y=ytrain)
    
    def lcb(x,posterior,data,beta=beta):
        x = x.reshape(-1,1)
        latent_dist = posterior.predict(x, train_data=data)
        predictive_dist = posterior.likelihood(latent_dist)

        mu = predictive_dist.mean()
        sigma = predictive_dist.stddev()

        return mu - beta * sigma
    
    a_lcb = lambda x: lcb(x, opt_posterior, D)[0]
    
    xn_s = []
    lcb_vals = []
    for x0 in x0_s:
        lbfgsb = ScipyBoundedMinimize(fun=a_lcb, method="l-bfgs-b")
        lower_bounds = jnp.ones_like(x0) * lower
        upper_bounds = jnp.ones_like(x0) * upper
        bounds = (lower_bounds, upper_bounds)
        lbfgsb_sol = lbfgsb.run(x0, bounds=bounds)
        xn = lbfgsb_sol.params
        xn_s.append(xn)
        lcb_vals.append(lbfgsb_sol.state.fun_val)
    
    xn_s = jnp.array(xn_s)
    lcb_vals = jnp.array(lcb_vals)
    return xn_s, lcb_vals


# %%
def plot_optimisation_of_acquisition(xn,yn,xtest,opt_posterior,xtrain,ytrain,beta=1.96):
    D = gpx.Dataset(X=xtrain, y=ytrain)
    
    def lcb(x,posterior,data,beta=beta):
        x = x.reshape(-1,1)
        latent_dist = posterior.predict(x, train_data=data)
        predictive_dist = posterior.likelihood(latent_dist)

        mu = predictive_dist.mean()
        sigma = predictive_dist.stddev()

        return mu - beta * sigma

    a_lcb = lambda x: lcb(x, opt_posterior, D)[0]
    
    plt.plot(x, y, "x", label="Observations", color=cols[0], alpha=0.5)
    plt.plot(xtest, lcb(xtest, opt_posterior, D), color=cols[1],label="LCB Acquisition function")
    plt.scatter(xn,a_lcb(xn),marker="o",color='green',label="minimizer of acquisition function")
    plt.legend()
    plt.title("LCB Acquistion function")
    plt.show()


# %% [markdown]
# # Black-box function
#
# We will use the [Gramacy and Lee (2012)](https://www.sfu.ca/~ssurjano/grlee12.html) one-dimensional test function:
# \begin{equation}
#     f(x) = \frac{\sin(10\pi x)}{2x} + (x-1)^{4}, \quad x\in[0.5,2.5]
# \end{equation}
#
# We plot this below:

# %%
f = lambda x: 0.5*(jnp.sin(10*jnp.pi*x)/x) + (x-1)**4
xtest = jnp.linspace(0.5,2.5,500).reshape(-1,1)
ytest = f(xtest)

plt.plot(xtest,ytest)
plt.grid()
plt.xlabel(r"$x$")
plt.title("Gramacy and Lee test function")
plt.show()

# %%
n = 3
x = jnp.linspace(0.5,2.5,n).reshape(-1,1)
y = f(x)

plt.plot(xtest,ytest,label='latent function')
plt.scatter(x,y, c='r',label='observations')
plt.legend()
plt.grid()
plt.xlabel(r"$x$")
plt.title("Initial observations")
plt.show()

# %%
N = 25
n_opt = 5

lower = 0.5
upper = 2.5

x0_s = jnp.linspace(lower,upper,n_opt).reshape(-1,1)

plot_bools = {
    'gp_training_loss': False,
    'gp_fit_and_acquisition': False,
    'optimisation_of_acquisition': False
}

for i in tqdm(range(N)):
    # split key
    key, _ = jr.split(key)
    
    # fit gp
    opt_posterior = fit_gp(x,y,key,plot_training_loss=plot_bools['gp_training_loss'])
    
    if plot_bools['gp_fit_and_acquisition']:
        # plot results of gp fit together with acquisition func
        plot_posterior_and_acquisition(xtest, opt_posterior, x, y)
    
    # optimise acquisition function
    xn_s, lcb_vals = optimise_acquisition(x0_s, opt_posterior, x, y, lower, upper)
    
    # take xn which has smallest lcb value
    xn = xn_s[lcb_vals.argmin()]
    # compute yn
    yn = f(xn)
    
    if plot_bools['optimisation_of_acquisition']:
        # plot results of optimisation of acquisition
        plot_optimisation_of_acquisition(xn,yn,xtest,opt_posterior,x,y)
    
    # update dataset
    x = jnp.vstack((x,xn))
    y = jnp.vstack((y,yn))

# %%
x[y.argmin()]

# %%
plot_posterior_and_acquisition(xtest,opt_posterior,x,y)

# %%
y_star = y.min()
x_star = x[y.argmin()]

# %%
y_dagger = ytest.min()
x_dagger = xtest[ytest.argmin()]

# %%
assert f(x_star) == y_star

# %%
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(xtest,ytest,label='latent function')
ax.scatter(x,y, c='r',label='observations')
ax.scatter(x_star,y_star,c='g',marker="o",label='BayesOPT minimiser')
ax.scatter(x_dagger,y_dagger,c='purple',marker="o",label='True minimiser')
ax.legend()
ax.grid()
ax.set_xlabel(r"$x$")
ax.set_title("Initial observations")
plt.show()

# %%
jnp.abs(y_dagger - y_star)

# %%
jnp.abs(x_dagger - x_star)
