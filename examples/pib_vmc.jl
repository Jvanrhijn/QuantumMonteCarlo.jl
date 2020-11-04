using Distributions
using DataStructures
using Random
using LinearAlgebra
using HDF5
using StatsBase

using QuantumMonteCarlo

# Force computation settings and import
const a = 1.0
const da = 1e-1

#include("forceutil.jl")

# DMC settings
τ = 1e-3
num_blocks = 10000
steps_per_block = trunc(Int64, 1/τ)
neq = 10

# Trial wave function
function ψpib(x::Array{Float64})
    max(0, (0.5a)^2 - x[1]^2)
end

function ψpib′(x::Array{Float64})
    max(0, (0.5a′)^2 - x[1]^2)
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

# Setting up the hamiltonian
hamiltonian(ψstatus, x) = -0.5*ψstatus.laplacian
hamiltonian_recompute(ψ, x) = -0.5*ψ.laplacian(x)

model = Model(
    hamiltonian,
    hamiltonian_recompute,
    ψtrial,
)

initial_conf = Float64[0.]
rng = MersenneTwister(123456)

### Actually run VMC
energies, errors = QuantumMonteCarlo.run_vmc!(
    initial_conf,
    model, 
    τ, 
    num_blocks, 
    steps_per_block;
    rng=rng, 
    neq=neq, 
    #outfile="test.hdf5", #ARGS[1],
    verbosity=:loud
);