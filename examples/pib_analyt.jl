using Distributions
using DataStructures
using Random
using LinearAlgebra
using HDF5
using StatsBase

using QuantumMonteCarlo

# Force computation settings and import
const a = 1
const da = 1e-2

# Setting up the hamiltonian
hamiltonian(ψstatus, x) = -0.5*ψstatus.laplacian
hamiltonian_recompute(ψ, x) = -0.5*ψ.laplacian(x)

include("forceutil_pib.jl")

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

ψtrial = WaveFunction(
    ψpib,
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
    "grad el" => gradel,
    "grad log psi" => grad_logpsi,
    "grad s" => (fwalker, model, eref) -> grads(fwalker, model, eref, τ),
    "grad t" => (fwalker, model, eref) -> gradt(fwalker, model, eref, τ),
)

rng = MersenneTwister(160224267)

# create "Fat" walkers
walkers = QuantumMonteCarlo.generate_walkers(nwalkers, ψtrial, rng, Uniform(-a/2, a/2), 1)


fat_walkers = [QuantumMonteCarlo.FatWalker(
    walker, 
    observables, 
    OrderedDict(
        "grad s" => CircularBuffer(lag),
        "grad t" => CircularBuffer(lag),
    ),
    [
        ("Local energy", "grad log psi"),
        ("Local energy", "grad s"),
        ("Local energy", "grad t"),
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
    brancher=stochastic_reconfiguration!,
    verbosity=:loud,
    branchtime=steps_per_block ÷ 10,
);