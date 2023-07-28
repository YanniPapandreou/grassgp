using LinearAlgebra, Distributions, Statistics, KernelFunctions, Random, Interpolations, Gridap, JLD
import Gridap: ∇

function eigenpairs(β, d)
    x_vals = range(0, stop=1, length=d)
    y_vals = range(0, stop=1, length=d)
    
    grid = []
    for x ∈ x_vals
        for y ∈ y_vals
            push!(grid, [x, y])
        end
    end
    dSq = d^2
    
    c = with_lengthscale(Matern12Kernel(), β)
    K = kernelmatrix(c, grid)
    
    spectrum = eigen(K, sortby = (x -> -real(x)))
    
    Φ = √(dSq) * spectrum.vectors
    lin_itps = map(i -> linear_interpolation((x_vals, y_vals), reshape(Φ[:,i],d,d)), 1:dSq)
    ϕ(x::Float64,y::Float64,i::Int) = lin_itps[i](x,y)
    λ = spectrum.values / d
    return ϕ, λ
end

function κ(x, p, ϕ, λ)
    dSq = length(p)
    log_κ = sum(p .* λ .* ϕ.(x[1], x[2], 1:dSq))
    return exp(log_κ)
end

function dκdp(x,p,ϕ,λ,j)
    dSq = length(p)
    log_κ = sum(p .* λ .* ϕ.(x[1], x[2], 1:dSq))
    return λ[j] * ϕ(x[1], x[2],j) * exp(log_κ)
end

function dudp(p, ϕ, λ_KL, κ, dκdp, x0, f, n=50, order=1)
    # set up mesh and spaces
    domain = (0,1,0,1)
    partition = (n,n)
    model = CartesianDiscreteModel(domain, partition)

    reffe = ReferenceFE(lagrangian,Float64,order)
    V0 = TestFESpace(model,reffe,conformity=:H1,dirichlet_tags="boundary")
    U = TrialFESpace(V0,x->0)

    degree = order + 1

    # solve for FEM solution uh of original problem (forward pass)
    Ω = Triangulation(model)
    dΩ = Measure(Ω,degree)

    kappa(x::VectorValue) = κ(x, p, ϕ, λ_KL)
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
    dSq = length(p)
    for j ∈ 1:dSq
        dkappa(x::VectorValue) = dκdp(x,p,ϕ,λ_KL,j)
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
d = 10
dSq = d^2
M = 300
ps = rand(Normal(), (dSq,M));

β = 1.0
ϕ, λ_KL = eigenpairs(β, d)
n_x = 12
n_y = 12

x_vals = range(0.05, stop=1-0.05, length=n_x)
y_vals = range(0.05, stop=1-0.05, length=n_y)

xs = []
for x ∈ x_vals
    for y ∈ y_vals
        push!(xs, [x, y])
    end
end

println("Starting to compute active subspaces.")
AS_results = []
for (i,x) ∈ enumerate(xs)
    @show i
    result = getAS(x, ps, ϕ, λ_KL, κ, dκdp, f)
    push!(AS_results, result)
end

save("matern_pde_data.jld", "AS_results", AS_results)
