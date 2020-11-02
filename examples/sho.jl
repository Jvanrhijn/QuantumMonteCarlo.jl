using Distributions
using DataStructures
using Random
using LinearAlgebra
using HDF5
using StatsBase
using ForwardDiff

using QuantumMonteCarlo

# Force computation settings and import
β = 0.378
a = 1.0
da = 1e-5
a′ = a + da

# Setting up the hamiltonian
hamiltonian(ψstatus, x) = -0.5*ψstatus.laplacian + 0.5 * a^2 * norm(x)^2 * ψstatus.value
hamiltonian_recompute(ψ, x) = -0.5*ψ.laplacian(x) + 0.5 * a^2 * norm(x)^2 * ψ.value(x)
hamiltonian′(ψstatus, x) = -0.5*ψstatus.laplacian + 0.5 * a′^2 * norm(x)^2 * ψstatus.value
hamiltonian_recompute′(ψ, x) = -0.5*ψ.laplacian(x) + 0.5 * a′^2 * norm(x)^2 * ψ.value(x)

include("forceutil.jl")

# DMC settings
#τ = 1e-2
τ = parse(Float64, ARGS[2])
nwalkers = 10
num_blocks = 1000
steps_per_block = trunc(Int64, 1/τ)
neq = 10
lag = trunc(Int64, 10*steps_per_block)
eref = 0.1 * (7β^2 + 1) / β
#fref = 0.1 * (7 - 1/a^2)
#eref = 0.625

# Trial wave function
function ψsho(x::AbstractArray)
    #exp(- a * norm(x)^2 / 4)
    1 / (1 + β*a*norm(x)^2)^2
end

function ψsho′(x::AbstractArray)
    a′ = a + da
    #exp(- a′ * norm(x)^2 / 4)
    1 / (1 + β*a′*norm(x)^2)^2
end

ψtrial = WaveFunction(
    ψsho,
    #x -> -2a*x*ψsho(x) / 4,
    #x -> ((-2*a*x[1] / 4)^2 - 2*a / 4)*ψsho(x),
    x -> -4*β*a*x / (β*a*norm(x)^2 + 1)^3,
    x -> 24(β*a)^2*norm(x)^2  / ((β*a)*norm(x)^2 + 1)^4 - 4a*β / (a*β*norm(x)^2 + 1)^3,
)

ψtrial′ = WaveFunction(
    ψsho′,
    #x -> -2(a + da)*x*ψsho′(x) / 4,
    #x -> ((-2*(a + da)*x[1] / 4)^2 - 2*(a + da) / 4)*ψsho′(x),
    x -> -4β*(a + da)*x / (β*(a + da)*norm(x)^2 + 1)^3,
    x -> 24β^2*(a + da)^2*norm(x)^2  / (β*(a + da)*norm(x)^2 + 1)^4 - 4β*(a + da) / (β*(a + da)*norm(x)^2 + 1)^3,
)

model = Model(
    hamiltonian,
    hamiltonian_recompute,
    ψtrial,
)

# TODO: fix the time spent in iterating over dicts
# Observables needed for force computation
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
    "grad s" => (fwalker, model, eref, xp) -> branching_factor_gradient(fwalker, model, eref, xp, ψtrial′, τ; warp=false),
    "grad g" => (fwalker, model, eref, xp) -> greens_function_gradient(fwalker, model, eref, xp, ψtrial′, τ; usepq=false, warp=false),
    "grad s (warp)" => (fwalker, model, eref, xp) -> branching_factor_gradient(fwalker, model, eref, xp, ψtrial′, τ; warp=true),
    "grad g (warp)" => (fwalker, model, eref, xp) -> greens_function_gradient(fwalker, model, eref, xp, ψtrial′, τ; usepq=false, warp=true),
    # S, T with p and q derivatives "grad s (p/q)" => (fwalker, model, eref, xp) -> branching_factor_gradient(fwalker, model, eref, xp, ψtrial′, τ; warp=false),
    "grad g (p/q)" => (fwalker, model, eref, xp) -> greens_function_gradient(fwalker, model, eref, xp, ψtrial′, τ; usepq=true, warp=false),
    "grad g (warp, p/q)" => (fwalker, model, eref, xp) -> greens_function_gradient(fwalker, model, eref, xp, ψtrial′, τ; usepq=true, warp=true),
    # Jacobians
    "grad log j" => (fwalker, model, eref, xp) -> jacobian_gradient_current(fwalker, model, eref, xp, ψtrial′, τ),
    "sum grad log j" => (fwalker, model, eref, xp) -> jacobian_gradient_previous(fwalker, model, eref, xp, ψtrial′, τ),
)

rng = MersenneTwister(160224267)

# create "Fat" walkers
walkers = QuantumMonteCarlo.generate_walkers(nwalkers, ψtrial, rng, Normal(0, 1), 1)


fat_walkers = [QuantumMonteCarlo.FatWalker(
    walker, 
    observables, 
    OrderedDict(
        "grad s" => CircularBuffer(lag),
        "grad g" => CircularBuffer(lag),
        "grad s (warp)" => CircularBuffer(lag),
        "grad g (warp)" => CircularBuffer(lag),
        "grad g (p/q)" => CircularBuffer(lag),
        "grad g (warp, p/q)" => CircularBuffer(lag),
        "sum grad log j" => CircularBuffer(lag),
    ),
    [
        ("Local energy", "grad log psi"),
        ("Local energy", "grad log psi (warp)"),
        ("Local energy", "grad s"),
        ("Local energy", "grad g"),
        ("Local energy", "grad s (warp)"),
        ("Local energy", "grad g (warp)"),
        ("Local energy", "grad g (p/q)"),
        ("Local energy", "grad g (warp, p/q)"),
        ("Local energy", "grad log j"),
        ("Local energy", "sum grad log j"),
    ]
    ) for walker in walkers
]

#fat_walkers = [QuantumMonteCarlo.FatWalker(walker) for walker in walkers]

### Actually run DMC
energies, errors = QuantumMonteCarlo.run_dmc!(
    model, 
    fat_walkers, 
    τ, 
    num_blocks, 
    steps_per_block, 
    eref,
    rng=rng, 
    neq=neq, 
    brancher=stochastic_reconfiguration_pyqmc!,
    #outfile="sho.hdf5",
    outfile=ARGS[1],
    verbosity=:loud,
    branchtime=steps_per_block ÷ 10,
);
