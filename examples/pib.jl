using Distributions
using Plots
using DataStructures
using Random
using LinearAlgebra
using HDF5
using StatsBase

using QuantumMonteCarlo

# Force computation settings and import
const a = 1.0
const da = 1e-5

include("forceutil.jl")

# DMC settings
τ = .5e-2
nwalkers = 100
num_blocks = 400
steps_per_block = Int64(1/τ)
neq = 10

# Trial wave function
function ψpib(x::Array{Float64})
    max(0, x[1].*(a .- x[1]))
end

function ψpib′(x::Array{Float64})
    max(0, x[1].*(a + da .- x[1]))
end

ψtrial = WaveFunction(
    ψpib,
    x -> a .- 2.0x,
    x -> -2.0,
)

ψtrial′ = WaveFunction(
    ψpib′,
    x -> a + da .- 2.0x,
    x -> -2.0,
)

# Setting up the hamiltonian
hamiltonian(ψstatus, x) = -0.5*ψstatus.laplacian
hamiltonian_recompute(ψ, x) = -0.5*ψ.laplacian(x)

model = Model(
    hamiltonian,
    hamiltonian_recompute,
    ψtrial,
)


# TODO: fix the time spent in iterating over dicts
# Observables needed for force computation
observables = OrderedDict(
    "Local energy" => local_energy,
    "Local energy (secondary)" => (fwalker, model, eref) -> local_energy_sec(fwalker, model, eref, ψtrial′),
    "ψ′" => (fwalker, model, eref) -> psi_sec(fwalker, model, eref, ψtrial′),
    "∇ψ′" => (fwalker, model, eref) -> gradpsi_sec(fwalker, model, eref, ψtrial′),
    "ψ′_old" => (fwalker, model, eref) -> psi_sec_old(fwalker, model, eref, ψtrial′),
    "∇ψ′_old" => (fwalker, model, eref) -> gradpsi_sec_old(fwalker, model, eref, ψtrial′),
    "grad el" => (fwalker, model, eref) -> gradel(fwalker, model, eref, ψtrial′),
    "grad el (warp)" => (fwalker, model, eref) -> gradel_warp(fwalker, model, eref, ψtrial′, τ),
    "grad log psi" => grad_logpsi,
    "grad log psi (warp)" => (fwalker, model, eref) -> grad_logpsi_warp(fwalker, model, eref, ψtrial′, τ),
    "grad s" => (fwalker, model, eref) -> grads(fwalker, model, eref, ψtrial′, τ),
    "grad t" => (fwalker, model, eref) -> gradt(fwalker, model, eref, ψtrial′, τ),
    "grad s (warp)" => (fwalker, model, eref) -> grads_warp(fwalker, model, eref, ψtrial′, τ),
    "grad t (warp)" => (fwalker, model, eref) -> gradt_warp(fwalker, model, eref, ψtrial′, τ),
    # These are placeholders, need to collect cutoff-ed versions as well
    "grad s (no cutoff)" => (fwalker, model, eref) -> grads(fwalker, model, eref, ψtrial′, τ),
    "grad t (no cutoff)" => (fwalker, model, eref) -> gradt(fwalker, model, eref, ψtrial′, τ),
    "grad s (warp, no cutoff)" => (fwalker, model, eref) -> grads_warp(fwalker, model, eref, ψtrial′, τ),
    "grad t (warp, no cutoff)" => (fwalker, model, eref) -> gradt_warp(fwalker, model, eref, ψtrial′, τ),
    "grad log j" => (fwalker, model, eref) -> gradj(fwalker, model, eref, τ),
    "sum grad log j" => (fwalker, model, eref) -> gradj(fwalker, model, eref, τ),
)

rng = MersenneTwister(0)

# create "Fat" walkers
walkers = QuantumMonteCarlo.generate_walkers(nwalkers, ψtrial, rng, Uniform(0., 1.), 1)


fat_walkers = [QuantumMonteCarlo.FatWalker(
    walker, 
    observables, 
    OrderedDict(
        "grad s" => CircularBuffer(steps_per_block),
        "grad t" => CircularBuffer(steps_per_block),
        "grad s (warp)" => CircularBuffer(steps_per_block),
        "grad t (warp)" => CircularBuffer(steps_per_block),
        "grad s (no cutoff)" => CircularBuffer(steps_per_block),
        "grad t (no cutoff)" => CircularBuffer(steps_per_block),
        "grad s (warp, no cutoff)" => CircularBuffer(steps_per_block),
        "grad t (warp, no cutoff)" => CircularBuffer(steps_per_block),
        "sum grad log j" => CircularBuffer(steps_per_block)
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
    ]
    ) for walker in walkers
]

#fat_walkers = [QuantumMonteCarlo.FatWalker(walker, OrderedDict()) for walker in walkers]

### Actually run DMC
energies, errors = QuantumMonteCarlo.run_dmc!(
    model, 
    fat_walkers, 
    τ, 
    num_blocks, 
    steps_per_block, 
    5.0; 
    rng=rng, 
    neq=neq, 
    outfile=ARGS[1],
    verbosity=:loud
);