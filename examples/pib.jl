using Distributions
using DataStructures
using Random
using LinearAlgebra
using HDF5
using StatsBase

using QuantumMonteCarlo

# Force computation settings and import
const a = 1
const da = 1e-5

# Setting up the hamiltonian
hamiltonian(ψstatus, x) = -0.5*ψstatus.laplacian
hamiltonian_recompute(ψ, x) = -0.5*ψ.laplacian(x)
hamiltonian′(ψstatus, x) = -0.5*ψstatus.laplacian
hamiltonian_recompute′(ψ, x) = -0.5*ψ.laplacian(x)

include("forceutil.jl")

# DMC settings
τ = 1e-2
nwalkers = 10
num_blocks = 100
steps_per_block = trunc(Int64, 1/τ)
neq = 10
lag = trunc(Int64, steps_per_block)
eref = 5.0/(2a)^2

# Trial wave function
function ψpib(x::Array{Float64})
    max(0, a^2 - x[1]^2)
end

function ψpib′(x::Array{Float64})
    a′ = a + da
    max(0, (a′)^2 - x[1]^2)
end

ψtrial = WaveFunction(
    ψpib,
    x -> -2x,
    x -> -2
)

ψtrial′ = WaveFunction(
    ψpib′,
    x -> -2x,
    x -> -2
)

model = Model(
    hamiltonian,
    hamiltonian_recompute,
    ψtrial,
)


# TODO: fix the time spent in iterating over dicts
# Observables needed for force computation
observables = OrderedDict(
    "ψ′" => (fwalker, model, eref, x) -> psi_sec(fwalker, model, eref, x, ψtrial′),
    "∇ψ′" => (fwalker, model, eref, x) -> gradpsi_sec(fwalker, model, eref, x, ψtrial′),
    "ψ′_old" => (fwalker, model, eref, x) -> psi_sec_old(fwalker, model, eref, x, ψtrial′),
    "∇ψ′_old" => (fwalker, model, eref, x) -> gradpsi_sec_old(fwalker, model, eref, x, ψtrial′),
    "Local energy" => local_energy,
    "Local energy (secondary)" => (fwalker, model, eref, x) -> local_energy_sec(fwalker, model, eref, x, ψtrial′),
    "grad el" => (fwalker, model, eref, x) -> gradel(fwalker, model, eref, x, ψtrial′),
    "grad el (warp)" => (fwalker, model, eref, x) -> gradel_warp(fwalker, model, eref, x, ψtrial′, τ),
    "grad log psi" => (fwalker, model, eref, x) -> grad_logpsi(fwalker, model, eref, x, ψtrial′),
    "grad log psi (warp)" => (fwalker, model, eref, x) -> grad_logpsi_warp(fwalker, model, eref, x, ψtrial′, τ),
    "grad s" => (fwalker, model, eref, x) -> grads(fwalker, model, eref, x, ψtrial′, τ),
    "grad t" => (fwalker, model, eref, x) -> gradt(fwalker, model, eref, x, ψtrial′, τ),
    "grad s (warp)" => (fwalker, model, eref, x) -> grads_warp(fwalker, model, eref, x, ψtrial′, τ),
    "grad t (warp)" => (fwalker, model, eref, x) -> gradt_warp(fwalker, model, eref, x, ψtrial′, τ),
    #These are placeholders, need to collect cutoff-ed versions as well
    "grad s (no cutoff)" => (fwalker, model, eref, x) -> grads(fwalker, model, eref, x, ψtrial′, τ),
    "grad t (no cutoff)" => (fwalker, model, eref, x) -> gradt(fwalker, model, eref, x, ψtrial′, τ),
    "grad s (warp, no cutoff)" => (fwalker, model, eref, x) -> grads_warp(fwalker, model, eref, x, ψtrial′, τ),
    "grad t (warp, no cutoff)" => (fwalker, model, eref, x) -> gradt_warp(fwalker, model, eref, x, ψtrial′, τ),
    "grad log j" => (fwalker, model, eref, x) -> gradj(fwalker, model, eref, x, τ),
    "sum grad log j" => (fwalker, model, eref, x) -> gradj(fwalker, model, eref, x, τ),
    "psi history" => psi_history,
    "psi history (secondary)" => (fwalker, model, eref, x) -> psi_history′(fwalker, model, eref, x, ψtrial′),
    "grad log psi squared old" => grad_logpsisquared_old,
)

#rng = MersenneTwister(160224267)
rng = MersenneTwister(9045943585439)

# create "Fat" walkers
walkers = QuantumMonteCarlo.generate_walkers(nwalkers, ψtrial, rng, Uniform(-a/2, a/2), 1)


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
    outfile="test.hdf5", #ARGS[1],
    brancher=stochastic_reconfiguration!,
    verbosity=:loud,
    branchtime=steps_per_block ÷ 10,
);