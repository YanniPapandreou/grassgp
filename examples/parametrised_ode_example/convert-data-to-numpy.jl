# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:percent
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Julia 1.9.1
#     language: julia
#     name: julia-1.9
# ---

# %%
using NPZ, JLD, Plots

# %%
d = 100
n_x = 40
xs = range(0.01,stop=0.99,length=n_x)

# %%
matern_ode_data = load("data/matern_ode_data.jld")

# %%
AS_results = matern_ode_data["AS_results"];

# %%
props = map(tup -> tup[3], AS_results);

# %%
plot(xs,props)

# %%
Ws = map(tup -> tup[1], AS_results);

Ws = reduce(hcat, Ws);

@assert size(Ws) == (d, n_x)

# %%
npzwrite("data/xs.npz", xs)

# %%
npzwrite("data/Ws.npz", Ws)

# %%
