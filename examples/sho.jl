using Distributions
using DataStructures
using Random
using LinearAlgebra
using HDF5
using StatsBase

using QuantumMonteCarlo

# Force computation settings and import
a = 1
da = 1e-5


# Setting up the hamiltonianj
hamiltonian(ψstatus, x) = -0.5*ψstatus.laplacian + 0.5 * a^2 * norm(x)^2*ψstatus.value
hamiltonian_recompute(ψ, x) = -0.5*ψ.laplacian(x) + 0.5 * a^2 * norm(x)^2*ψ.value(x)
hamiltonian′(ψstatus, x) = -0.5*ψstatus.laplacian + 0.5 * (a + da)^2 * norm(x)^2*ψstatus.value
hamiltonian_recompute′(ψ, x) = -0.5*ψ.laplacian(x) + 0.5 * (a + da)^2 * norm(x)^2*ψ.value(x)

include("forceutil.jl")

# DMC settings
τ = 1e-2
nwalkers = 40
num_blocks = 100
steps_per_block = trunc(Int64, 1/τ)
neq = 10
lag = trunc(Int64, 10*steps_per_block)
eref = 0.625

# Trial wave function
function ψsho(x::Array{Float64})
    exp(- a * norm(x)^2)
end

function ψsho′(x::Array{Float64})
    a′ = a + da
    exp(- a′ * norm(x)^2)
end

ψtrial = WaveFunction(
    ψsho,
    x -> QuantumMonteCarlo.gradient_fd(ψsho, x),
    x -> QuantumMonteCarlo.laplacian_fd(ψsho, x),
)

ψtrial′ = WaveFunction(
    ψsho′,
    x -> QuantumMonteCarlo.gradient_fd(ψsho′, x),
    x -> QuantumMonteCarlo.laplacian_fd(ψsho′, x),
)

model = Model(
    hamiltonian,
    hamiltonian_recompute,
    ψtrial,
)


# TODO: fix the time spent in iterating over dicts
# Observables needed for force computation
observables = OrderedDict(
    "ψ′" => (fwalker, model, eref, xp) -> psi_sec(fwalker, model, eref, xp, ψtrial′),
    "∇ψ′" => (fwalker, model, eref, xp) -> gradpsi_sec(fwalker, model, eref, xp, ψtrial′),
    "ψ′_old" => (fwalker, model, eref, xp) -> psi_sec_old(fwalker, model, eref, xp, ψtrial′),
    "∇ψ′_old" => (fwalker, model, eref, xp) -> gradpsi_sec_old(fwalker, model, eref, xp, ψtrial′),
    "Local energy" => local_energy,
    "Local energy (secondary)" => (fwalker, model, eref, xp) -> local_energy_sec(fwalker, model, eref, xp, ψtrial′),
    "grad el" => (fwalker, model, eref, xp) -> gradel(fwalker, model, eref, xp, ψtrial′),
    "grad el (warp)" => (fwalker, model, eref, xp) -> gradel_warp(fwalker, model, eref, xp, ψtrial′, τ),
    "grad log psi" => (fwalker, model, eref, xp) -> grad_logpsi(fwalker, model, eref, xp, ψtrial′),
    "grad log psi (warp)" => (fwalker, model, eref, xp) -> grad_logpsi_warp(fwalker, model, eref, xp, ψtrial′, τ),
    "grad s" => (fwalker, model, eref, xp) -> grads(fwalker, model, eref, xp, ψtrial′, τ),
    "grad t" => (fwalker, model, eref, xp) -> gradt(fwalker, model, eref, xp, ψtrial′, τ),
    "grad s (warp)" => (fwalker, model, eref, xp) -> grads_warp(fwalker, model, eref, xp, ψtrial′, τ),
    "grad t (warp)" => (fwalker, model, eref, xp) -> gradt_warp(fwalker, model, eref, xp, ψtrial′, τ),
    #These are placeholders, need to collect cutoff-ed versions as well
    "grad s (no cutoff)" => (fwalker, model, eref, xp) -> grads(fwalker, model, eref, xp, ψtrial′, τ),
    "grad t (no cutoff)" => (fwalker, model, eref, xp) -> gradt(fwalker, model, eref, xp, ψtrial′, τ),
    "grad s (warp, no cutoff)" => (fwalker, model, eref, xp) -> grads_warp(fwalker, model, eref, xp, ψtrial′, τ),
    "grad t (warp, no cutoff)" => (fwalker, model, eref, xp) -> gradt_warp(fwalker, model, eref, xp, ψtrial′, τ),
    "grad log j" => (fwalker, model, eref, xp) -> gradj(fwalker, model, eref, xp, τ),
    "sum grad log j" => (fwalker, model, eref, xp) -> gradj(fwalker, model, eref, xp, τ),
    "psi history" => psi_history,
    "psi history (secondary)" => (fwalker, model, eref, xp) -> psi_history′(fwalker, model, eref, xp, ψtrial′),
    "grad log psi squared old" => grad_logpsisquared_old,
)

rng = MersenneTwister(160224267)

# create "Fat" walkers
walkers = QuantumMonteCarlo.generate_walkers(nwalkers, ψtrial, rng, Normal(0, 1), 1)


fat_walkers = [QuantumMonteCarlo.FatWalker(
    walker, 
    observables, 
    OrderedDict(
        "grad s" => CircularBuffer(lag),
        "grad t" => CircularBuffer(lag),
        "grad s (warp)" => CircularBuffer(lag),
        "grad t (warp)" => CircularBuffer(lag),
        "grad s (no cutoff)" => CircularBuffer(lag),
        "grad t (no cutoff)" => CircularBuffer(lag),
        "grad s (warp, no cutoff)" => CircularBuffer(lag),
        "grad t (warp, no cutoff)" => CircularBuffer(lag),
        "sum grad log j" => CircularBuffer(lag),
        "psi history" => CircularBuffer(lag),
        "psi history (secondary)" => CircularBuffer(lag),
    ),
    [
        ("Local energy", "grad log psi"),
        ("Local energy", "grad log psi (warp)"),
        ("Local energy", "grad s"),
        ("Local energy", "grad t"),
        ("Local energy", "grad s (warp)"),
        ("Local energy", "grad t (warp)"),
        ("Local energy", "grad s (no cutoff)"),
        ("Local energy", "grad t (no cutoff)"),
        ("Local energy", "grad s (warp, no cutoff)"),
        ("Local energy", "grad t (warp, no cutoff)"),
        ("Local energy", "grad log j"),
        ("Local energy", "sum grad log j"),
        ("Local energy", "grad log psi squared old")
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
    #brancher=optimal_stochastic_reconfiguration!,
    outfile="sho.hdf5", #ARGS[1],
    verbosity=:loud,
    branchtime=steps_per_block ÷ 10,
    #branchtime=1,
);
