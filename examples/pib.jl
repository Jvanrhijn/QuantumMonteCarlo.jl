using Distributions
using DataStructures
using Random
using LinearAlgebra
using HDF5
using StatsBase

using QuantumMonteCarlo

# Force computation settings and import
a = 1.0
da = 1e-5

# Setting up the hamiltonian
hamiltonian(ψstatus, x) = -0.5*ψstatus.laplacian
hamiltonian_recompute(ψ, x) = -0.5*ψ.laplacian(x)
hamiltonian′(ψstatus, x) = -0.5*ψstatus.laplacian
hamiltonian_recompute′(ψ, x) = -0.5*ψ.laplacian(x)

include("forceutil.jl")

# DMC setting
τ = .1e-1
nwalkers = 10
num_blocks = 700
steps_per_block = trunc(Int64, 1/τ)
neq = 10
lag = trunc(Int64, steps_per_block)
eref = 5.0/(2a)^2

# Trial wave function
#function ψpib(x::Array{Float64})
function ψpib(x::AbstractArray)
    a^2 - x[1]^2
end

#function ψpib′(x::Array{Float64})
function ψpib′(x::AbstractArray)
    a′ = a + da
    (a′)^2 - x[1]^2
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
    # Local energy
    "Local energy" => local_energy,
    # Gradients of local energy
    "grad el" => (fwalker, model, eref, xp) -> gradel(fwalker, model, eref, xp, ψtrial′, τ; warp=false),
    "grad el (warp)" => (fwalker, model, eref, xp) -> gradel(fwalker, model, eref, xp, ψtrial′, τ; warp=true),
    # Gradients of log(ψ)
    "grad log psi" => (fwalker, model, eref, xp) -> grad_logpsi(fwalker, model, eref, xp, ψtrial′),
    "grad log psi (warp)" => (fwalker, model, eref, xp) -> grad_logpsi_warp(fwalker, model, eref, xp, ψtrial′, τ),
    # S, T with and without warp
    "grad s" => (fwalker, model, eref, xp) -> grads(fwalker, model, eref, xp, ψtrial′, τ; warp=false),
    "grad t" => (fwalker, model, eref, xp) -> gradt(fwalker, model, eref, xp, ψtrial′, τ; usepq=false, warp=false),
    "grad s (warp)" => (fwalker, model, eref, xp) -> grads(fwalker, model, eref, xp, ψtrial′, τ; warp=true),
    "grad t (warp)" => (fwalker, model, eref, xp) -> gradt(fwalker, model, eref, xp, ψtrial′, τ; usepq=false, warp=true),
    # S, T with p and q derivatives
    "grad s (p/q)" => (fwalker, model, eref, xp) -> grads(fwalker, model, eref, xp, ψtrial′, τ; warp=false),
    "grad t (p/q)" => (fwalker, model, eref, xp) -> gradt(fwalker, model, eref, xp, ψtrial′, τ; usepq=true, warp=false),
    "grad s (warp, p/q)" => (fwalker, model, eref, xp) -> grads(fwalker, model, eref, xp, ψtrial′, τ; warp=true),
    "grad t (warp, p/q)" => (fwalker, model, eref, xp) -> gradt(fwalker, model, eref, xp, ψtrial′, τ; usepq=true, warp=true),
    # Jacobians
    "grad log j" => (fwalker, model, eref, xp) -> gradj_last(fwalker, model, eref, xp, ψtrial′, τ),
    "sum grad log j" => (fwalker, model, eref, xp) -> gradj(fwalker, model, eref, xp, ψtrial′, τ),
    # old trial fn values
    "grad log psi hist" => (fwalker, model, eref, xp) -> grad_logpsi(fwalker, model, eref, xp, ψtrial′),
    "grad log psi hist (warp)" => (fwalker, model, eref, xp) -> grad_logpsi_warp(fwalker, model, eref, xp, ψtrial′, τ),
    "grad log psi old" => (fwalker, model, eref, xp) -> grad_logpsi_old(fwalker, model, eref, xp),
    "grad log psi old (warp)" => (fwalker, model, eref, xp) -> grad_logpsi_old_warp(fwalker, model, eref, xp),
    # pulay force warp correction for exact force
    "pulay warp correction exact" => (fwalker, model, eref, xp) -> pulay_force_warp_correction_exact(fwalker, model, eref, xp, ψtrial′, τ),
)

rng = MersenneTwister(16224267)

# create "Fat" walkers
walkers = QuantumMonteCarlo.generate_walkers(nwalkers, ψtrial, rng, Uniform(-a, a), 1)

fat_walkers = [QuantumMonteCarlo.FatWalker(
    walker, 
    observables, 
    OrderedDict(
        "grad s" => CircularBuffer(lag),
        "grad t" => CircularBuffer(lag),
        "grad s (warp)" => CircularBuffer(lag),
        "grad t (warp)" => CircularBuffer(lag),
        "grad s (p/q)" => CircularBuffer(lag),
        "grad t (p/q)" => CircularBuffer(lag),
        "grad s (warp, p/q)" => CircularBuffer(lag),
        "grad t (warp, p/q)" => CircularBuffer(lag),
        "sum grad log j" => CircularBuffer(lag),
        "grad log psi hist" => CircularBuffer(lag),
        "grad log psi hist (warp)" => CircularBuffer(lag),
    ),
    [
        ("Local energy", "grad log psi"),
        ("Local energy", "pulay warp correction exact"),
        ("Local energy", "grad log psi (warp)"),
        ("Local energy", "grad s"),
        ("Local energy", "grad t"),
        ("Local energy", "grad s (warp)"),
        ("Local energy", "grad t (warp)"),
        ("Local energy", "grad s (p/q)"),
        ("Local energy", "grad t (p/q)"),
        ("Local energy", "grad s (warp, p/q)"),
        ("Local energy", "grad t (warp, p/q)"),
        ("Local energy", "grad log j"),
        ("Local energy", "sum grad log j"),
        ("Local energy", "grad log psi old"),
        ("Local energy", "grad log psi old (warp)"),
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
    #brancher=stochastic_reconfiguration!,
    #brancher=no_brancher!,
    outfile="pib.hdf5", #ARGS[1],
    verbosity=:loud,
    branchtime=steps_per_block ÷ 10,
    #branchtime=5,
);
