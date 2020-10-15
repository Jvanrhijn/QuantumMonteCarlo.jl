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

include("forceutil_pib.jl")

# DMC settings
τ = 2e-2
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
    max(0, a′^2 - x[1]^2)
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
    "Local energy" => local_energy,
    "grad el" => (fwalker, model, eref) -> gradel(fwalker, model, eref, ψtrial′),
    "grad el (warp)" => (fwalker, model, eref) -> gradel_warp(fwalker, model, eref, ψtrial′),
    "grad log psi" => (fwalker, model, eref) -> grad_logpsi(fwalker, model, eref, ψtrial′),
    "grad log psi (warp)" => (fwalker, model, eref) -> grad_logpsi_warp(fwalker, model, eref, ψtrial′),
    "grad s" => (fwalker, model, eref) -> grads(fwalker, model, eref, ψtrial′, τ),
    "grad s (warp)" => (fwalker, model, eref) -> grads_warp(fwalker, model, eref, ψtrial′, τ),
    "grad t" => (fwalker, model, eref) -> gradt(fwalker, model, eref, ψtrial′, τ),
    "grad t (warp)" => (fwalker, model, eref) -> gradt_warp(fwalker, model, eref, ψtrial′, τ),
    "grad j" => grad_jac,
    "grad sum j" => grad_jac,
)

rng = MersenneTwister(160224267)

# create "Fat" walkers
walkers = QuantumMonteCarlo.generate_walkers(nwalkers, ψtrial, rng, Uniform(-a, a), 1)


fat_walkers = [QuantumMonteCarlo.FatWalker(
    walker, 
    observables, 
    OrderedDict(
        "grad s" => CircularBuffer(lag),
        "grad s (warp)" => CircularBuffer(lag),
        "grad t" => CircularBuffer(lag),
        "grad t (warp)" => CircularBuffer(lag),
        "grad sum j" => CircularBuffer(lag),
    ),
    [
        ("Local energy", "grad log psi"),
        ("Local energy", "grad log psi (warp)"),
        ("Local energy", "grad s"),
        ("Local energy", "grad s (warp)"),
        ("Local energy", "grad t"),
        ("Local energy", "grad t (warp)"),
        ("Local energy", "grad j"),
        ("Local energy", "grad sum j"),
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
    outfile="pib_analyt.hdf5", #ARGS[1],
    #brancher=stochastic_reconfiguration!,
    brancher=stochastic_reconfiguration!,
    verbosity=:loud,
    branchtime=steps_per_block ÷ 10,
    #branchtime=1,
);