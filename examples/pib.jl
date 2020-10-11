using Distributions
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
τ = 1e-3
nwalkers = 25
num_blocks = 1000
steps_per_block = trunc(Int64, 1/10τ)
neq = 10
lag = trunc(Int64, steps_per_block)
eref = 5.0/a^2

# Trial wave function
function ψpib(x::Array{Float64})
    #max(0, sin(pi*x[1])/a)
    #max(0, 4*x[1].*(a .- x[1]) + sin(pi*x[1]/a))
    #max(0, x[1]*(a - x[1]))
    #max(0, (1 + x[1])*sin(π*x[1]/a))
    max(0, (0.5a)^2 - x[1]^2)
end

function ψpib′(x::Array{Float64})
    #max(0, sin(pi*x[1]/(a+da)))
    #max(0, 4*x[1].*(a + da .- x[1]) + sin(pi*x[1]/(a+da)))
    #max(0, (x[1] + da/2)*(a + da/2 - x[1]))
    #max(0, (1 + x[1])*sin(π*x[1]/(a + da)))
    a′ = a + da
    max(0, (0.5a′)^2 - x[1]^2)
end

ψtrial = WaveFunction(
    ψpib,
    #x -> π/a*cos.(π*x/a).*(1 .+ x) + sin.(π*x/a),
    #x -> -π*(π*(x[1] + 1)*sin(π*x[1]/a) - 2a*cos(π*x[1]/a))/a^2
    #x -> pi*cos.(pi*x/a)/a,
    #x -> -(pi/a)^2*sin(pi*x[1]/a)
    #x -> 4*(a .- 2x) + π/a*cos.(pi*x/a),
    #x -> -8.0 - (π/a)^2*sin(pi*x[1]/a)
    #x -> QuantumMonteCarlo.gradient_fd(ψpib, x),
    #x -> QuantumMonteCarlo.laplacian_fd(ψpib, x)
    x -> -2x,
    x -> -2
)

ψtrial′ = WaveFunction(
    ψpib′,
    #x -> π/(a+da)*cos.(π*x/(a+da)).*(1 .+ x) + sin.(π*x/(a+da)),
    #x -> -π*(π*(x[1] + 1)*sin(π*x[1]/(a+da)) - 2a*cos(π*x[1]/(a+da)))/(a+da)^2
    #x -> pi*cos.(pi*x/(a + da))/(a + da),
    #x -> -(pi/(a + da))^2*sin(pi*x[1]/(a + da))
    #x -> 4*(a + da .- 2x) + π/(a + da)*cos.(pi*x/(a + da)),
    #x -> -8.0 - (π/(a+da))^2*sin(pi*x[1]/(a + da))
    #x -> QuantumMonteCarlo.gradient_fd(ψpib′, x),
    #x -> QuantumMonteCarlo.laplacian_fd(ψpib′, x),
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


# TODO: fix the time spent in iterating over dicts
# Observables needed for force computation
observables = OrderedDict(
    "ψ′" => (fwalker, model, eref) -> psi_sec(fwalker, model, eref, ψtrial′),
    "∇ψ′" => (fwalker, model, eref) -> gradpsi_sec(fwalker, model, eref, ψtrial′),
    "ψ′_old" => (fwalker, model, eref) -> psi_sec_old(fwalker, model, eref, ψtrial′),
    "∇ψ′_old" => (fwalker, model, eref) -> gradpsi_sec_old(fwalker, model, eref, ψtrial′),
    "Local energy" => local_energy,
    "Local energy (secondary)" => (fwalker, model, eref) -> local_energy_sec(fwalker, model, eref, ψtrial′),
    "grad el" => (fwalker, model, eref) -> gradel(fwalker, model, eref, ψtrial′),
    "grad el (warp)" => (fwalker, model, eref) -> gradel_warp(fwalker, model, eref, ψtrial′, τ),
    "grad log psi" => (fwalker, model, eref) -> grad_logpsi(fwalker, model, eref, ψtrial′),
    "grad log psi (warp)" => (fwalker, model, eref) -> grad_logpsi_warp(fwalker, model, eref, ψtrial′, τ),
    "grad s" => (fwalker, model, eref) -> grads(fwalker, model, eref, ψtrial′, τ),
    "grad t" => (fwalker, model, eref) -> gradt(fwalker, model, eref, ψtrial′, τ),
    "grad s (warp)" => (fwalker, model, eref) -> grads_warp(fwalker, model, eref, ψtrial′, τ),
    "grad t (warp)" => (fwalker, model, eref) -> gradt_warp(fwalker, model, eref, ψtrial′, τ),
    #These are placeholders, need to collect cutoff-ed versions as well
    "grad s (no cutoff)" => (fwalker, model, eref) -> grads(fwalker, model, eref, ψtrial′, τ),
    "grad t (no cutoff)" => (fwalker, model, eref) -> gradt(fwalker, model, eref, ψtrial′, τ),
    "grad s (warp, no cutoff)" => (fwalker, model, eref) -> grads_warp(fwalker, model, eref, ψtrial′, τ),
    "grad t (warp, no cutoff)" => (fwalker, model, eref) -> gradt_warp(fwalker, model, eref, ψtrial′, τ),
    "grad log j" => (fwalker, model, eref) -> gradj(fwalker, model, eref, τ),
    "sum grad log j" => (fwalker, model, eref) -> gradj(fwalker, model, eref, τ),
    "psi history" => psi_history,
    "psi history (secondary)" => (fwalker, model, eref) -> psi_history′(fwalker, model, eref, ψtrial′),
    "grad log psi squared old" => grad_logpsisquared_old,
)

rng = MersenneTwister(160224267)

# create "Fat" walkers
walkers = QuantumMonteCarlo.generate_walkers(nwalkers, ψtrial, rng, Uniform(-0.5a, 0.5a), 1)


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
    verbosity=:loud
);