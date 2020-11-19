using Distributions
using DataStructures
using Random
using LinearAlgebra
using HDF5
using StatsBase

using QuantumMonteCarlo

# Force computation settings and import
a = 1.001
da = 1e-5

# Setting up the hamiltonian
hamiltonian(ψstatus, x) = -0.5*ψstatus.laplacian
hamiltonian_recompute(ψ, x) = -0.5*ψ.laplacian(x)
hamiltonian′(ψstatus, x) = -0.5*ψstatus.laplacian
hamiltonian_recompute′(ψ, x) = -0.5*ψ.laplacian(x)

include("forceutil_vmc.jl")

# VMC settings
τ = 1e-2
nwalkers = 1
num_blocks = 4*16000
steps_per_block = max(100, trunc(Int64, 1/τ))
neq = num_blocks ÷ 10
lag = trunc(Int64, steps_per_block)

α(a) = a*cosh(1)
β(a) = a*sinh(1)

# Trial wave function
function ψpib(x::AbstractArray)
    r = (x[1] / α(a))^2 + (x[2] / β(a))^2
    return 1 - r
end

function ψpib′(x::AbstractArray)
    a′ = a + da
    r = (x[1] / α(a′))^2 + (x[2] / β(a′))^2
    return 1 - r
end

ψtrial = WaveFunction(
    ψpib,
    x -> -2 * [x[1]/α(a)^2, x[2]/β(a)^2],
    x -> -2(1/α(a)^2 + 1/β(a)^2),
)

ψtrial′ = WaveFunction(
    ψpib′,
    x -> -2 * [x[1]/α(a + da)^2, x[2]/β(a + da)^2],
    x -> -2(1/α(a + da)^2 + 1/β(a + da)^2),
)

model = Model(
    hamiltonian,
    hamiltonian_recompute,
    ψtrial,
)

observables = OrderedDict(
    # Local energy
    "Local energy" => local_energy,
    # Gradients of local energy
    "grad el" => (fwalker, model, eref, xp) -> local_energy_gradient(fwalker, model, eref, xp, ψtrial′, τ; warp=false),
    "grad el (warp)" => (fwalker, model, eref, xp) -> local_energy_gradient(fwalker, model, eref, xp, ψtrial′, τ; warp=true),
    # Gradients of log(ψ)
    "grad log psi" => (fwalker, model, eref, xp) -> log_psi_gradient(fwalker, model, eref, xp, ψtrial′, τ; warp=false),
    "grad log psi (warp)" => (fwalker, model, eref, xp) -> log_psi_gradient(fwalker, model, eref, xp, ψtrial′, τ; warp=true),
    # S, T with and without warp
    "grad g" => (fwalker, model, eref, xp) -> greens_function_gradient(fwalker, model, eref, xp, ψtrial′, τ; usepq=false, warp=false),
    "grad g (warp)" => (fwalker, model, eref, xp) -> greens_function_gradient(fwalker, model, eref, xp, ψtrial′, τ; usepq=false, warp=true),
    # S, T with p and q derivatives
    "grad g (p/q)" => (fwalker, model, eref, xp) -> greens_function_gradient(fwalker, model, eref, xp, ψtrial′, τ; usepq=true, warp=false),
    "grad g (warp, p/q)" => (fwalker, model, eref, xp) -> greens_function_gradient(fwalker, model, eref, xp, ψtrial′, τ; usepq=true, warp=true),
    # Jacobians
    "grad log j" => (fwalker, model, eref, xp) -> jacobian_gradient_current(fwalker, model, eref, xp, ψtrial′, τ),
    "sum grad log j" => (fwalker, model, eref, xp) -> jacobian_gradient_previous(fwalker, model, eref, xp, ψtrial′, τ),
    # Jacobians approximate
    "grad log j approx" => (fwalker, model, eref, xp) -> jacobian_gradient_current_approx(fwalker, model, eref, xp, ψtrial′, τ),
    "sum grad log j approx" => (fwalker, model, eref, xp) -> jacobian_gradient_previous_approx(fwalker, model, eref, xp, ψtrial′, τ),
)

rng = MersenneTwister(16224267)

# create "Fat" walkers
walkers = QuantumMonteCarlo.generate_walkers(nwalkers, ψtrial, rng, Uniform(-0.5, 0.5), 2)

fat_walkers = [QuantumMonteCarlo.FatWalker(
    walker, 
    observables, 
    OrderedDict(
        "grad g" => CircularBuffer(lag),
        "grad g (warp)" => CircularBuffer(lag),
        "grad g (p/q)" => CircularBuffer(lag),
        "grad g (warp, p/q)" => CircularBuffer(lag),
        "sum grad log j" => CircularBuffer(lag),
    ),
    [
        ("Local energy", "grad log psi"),
        ("Local energy", "grad log psi (warp)"),
        ("Local energy", "grad g"),
        ("Local energy", "grad g (warp)"),
        ("Local energy", "grad g (p/q)"),
        ("Local energy", "grad g (warp, p/q)"),
        ("Local energy", "grad log j"),
        ("Local energy", "sum grad log j"),
        ("Local energy", "grad log j approx"),
        ("Local energy", "sum grad log j approx"),
    ]
    ) for walker in walkers
]

#fat_walkers = [QuantumMonteCarlo.FatWalker(walker) for walker in walkers]

energies, errors = QuantumMonteCarlo.run_vmc!(
    model, 
    fat_walkers, 
    τ, 
    num_blocks, 
    steps_per_block, 
    rng=rng, 
    neq=neq, 
    outfile="ellipse_vmc.hdf5",
    verbosity=:loud,
);