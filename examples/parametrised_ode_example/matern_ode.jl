using LinearAlgebra, Distributions, Statistics, Plots, KernelFunctions, Random, Interpolations, ForwardDiff, Gridap, NPZ, JLD
import Gridap: ∇

function eigenpairs(β,d)
    xs = range(0, stop=1, length=d)
    
    # form kernel and kernel mat
    c = with_lengthscale(Matern12Kernel(), β)
    K = kernelmatrix(c, xs)

    # get spectrum
    spectrum = eigen(K, sortby = (x -> -real(x)))
    
    # form eigenpairs
    Φ = √(d) * spectrum.vectors
    # ϕ(x::Float64,i::Int) = linear_interpolation(xs, Φ[:,i])(x)
    lin_itps = map(i -> linear_interpolations(xs, Φ[:,i]))(x)
    ϕ(x::Float64,i::Int) = lin_itps[i](x)
    λ = spectrum.values / d
    return ϕ, λ
end

function κ(x, p, ϕ, λ)
    d = length(p)
    # compute log_κ and κ
    log_κ = sum(p .* λ .* ϕ.(x, 1:d))
    return exp(log_κ)
end

function dκdp(x,p,ϕ,λ,j)
    d = length(p)
    log_κ = sum(p .* λ .* ϕ.(x, 1:d))
    return λ[j] * ϕ(x,j) * exp(log_κ)
end

function dudp(p, ϕ, λ_KL, κ, dκdp, x0, f, n=50, order=1)
    # set up mesh and spaces
    domain = (0,1)
    partition = (n)
    model = CartesianDiscreteModel(domain, partition)

    reffe = ReferenceFE(lagrangian,Float64,order)
    V0 = TestFESpace(model,reffe,conformity=:H1,dirichlet_tags="boundary")
    U = TrialFESpace(V0,x->0)

    degree = order + 1

    # solve for FEM solution uh of original problem (forward pass)
    Ω = Triangulation(model)
    dΩ = Measure(Ω,degree)

    kappa(x::VectorValue) = κ(x[1],p, ϕ, λ_KL)
    a(u,v) = ∫( kappa * ∇(v)⊙∇(u) )*dΩ
    b(v) = ∫( v*f )*dΩ

    op = AffineFEOperator(a,b,U,V0)
    uh = solve(op)

    # solve for adjoint variable λ (back pass)
    δ = DiracDelta(model, Point(x0))
    b_λ(v) = -1*δ(v)
    
    op_λ = AffineFEOperator(a,b_λ,U,V0)
    λ = solve(op_λ)

    # compute derivative for each parameter and return and return
    grads = []
    d = length(p)
    for j ∈ 1:d
        dkappa(x::VectorValue) = dκdp(x[1],p,ϕ,λ_KL,j)
        sens = sum( ∫( dkappa * ∇(λ)⋅∇(uh) )*dΩ )
        push!(grads, sens)
    end
    
    return grads
end

function getAS(x0, ps, ϕ, λ_KL, κ, dκdp, f)
    # compute gradients at each p in ps
    Grads = map(p -> dudp(p, ϕ, λ_KL, κ, dκdp, x0, f), eachcol(ps))

    # take outer product of each grad with itself
    C_x = map(grad -> grad * grad', Grads)

    # compute mean of outer products
    C_hat = Statistics.mean(C_x)

    # eigen decomposition of C_hat
    spectrum = eigen(C_hat)
    λ = spectrum.values

    # compute proportion of spectrum explained by max eigenvector
    λ_max = max(λ...)
    prop = λ_max / sum(abs,λ)

    # get AS as leading eigenvector
    W = spectrum.vectors[:, argmax(λ)]
    @assert W' * W ≈ 1.0

    return W, λ, prop
end

f(x) = 1.0
Random.seed!(42)
d = 100
M = 300
ps = rand(Normal(), (d,M));
β = 1.0
ϕ, λ_KL = eigenpairs(β, d)
n_x = 40
xs = range(0.01,stop=0.99,length=n_x)

println("Starting to compute active subspaces.")
AS_results = []
for (i,x) ∈ enumerate(xs)
    @show i
    result = getAS(x, ps, ϕ, λ_KL, κ, dκdp, f)
    push!(AS_results, result)
end

save("matern_ode_data.jld", "AS_results", AS_results)
