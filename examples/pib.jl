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
nwalkers = 100
num_blocks = 400
steps_per_block = 100
neq = 10
τ = 1e-2

# Trial wave function
ψpib(x) = max(0, x[1].*(a .- x[1]))
ψpib′(x) = max(0, x[1].*(a + da .- x[1]))

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


# Observables needed for force computation
observables = OrderedDict(
    "ψ′" => (fwalker, model, eref) -> psi_sec(fwalker, model, eref, ψtrial′),
    "∇ψ′" => (fwalker, model, eref) -> gradpsi_sec(fwalker, model, eref, ψtrial′),
    "ψ′_old" => (fwalker, model, eref) -> psi_sec(fwalker, model, eref, ψtrial′),
    "∇ψ′_old" => (fwalker, model, eref) -> gradpsi_sec(fwalker, model, eref, ψtrial′),
    "Local energy" => local_energy,
    "grad el" => (fwalker, model, eref) -> gradel(fwalker, model, eref, ψtrial′),
    "grad el (warp)" => (fwalker, model, eref) -> gradel_warp(fwalker, model, eref, ψtrial′),
    "grad log psi" => grad_logpsi,
    "grad log psi (warp)" => (fwalker, model, eref) -> grad_logpsi_warp(fwalker, model, eref, ψtrial′),
    "grad s" => (fwalker, model, eref) -> grads(fwalker, model, eref, ψtrial′, τ),
    "grad t" => (fwalker, model, eref) -> gradt(fwalker, model, eref, ψtrial′, τ),
    "grad s (warp)" => (fwalker, model, eref) -> grads_warp(fwalker, model, eref, ψtrial′, τ),
    "grad t (warp)" => (fwalker, model, eref) -> gradt_warp(fwalker, model, eref, ψtrial′, τ),
    "grad log j" => gradj,
    "sum grad log j" => gradj,
)

rng = MersenneTwister(0)

# create "Fat" walkers
walkers = QuantumMonteCarlo.generate_walkers(nwalkers, ψtrial, rng, Uniform(0., 1.), (1, 1))

#fat_walkers = [QuantumMonteCarlo.FatWalker(walker, OrderedDict()) for walker in walkers]
fat_walkers = [QuantumMonteCarlo.FatWalker(
    walker, 
    observables, 
    OrderedDict(
        "grad s" => CircularBuffer(steps_per_block),
        "grad t" => CircularBuffer(steps_per_block),
        "grad s (warp)" => CircularBuffer(steps_per_block),
        "grad t (warp)" => CircularBuffer(steps_per_block),
        "sum grad log j" => CircularBuffer(steps_per_block)
    ),
    [
        ("Local energy", "grad log psi"),
        ("Local energy", "grad log psi (warp)"),
        ("Local energy", "grad s"),
        ("Local energy", "grad t"),
        ("Local energy", "grad s (warp)"),
        ("Local energy", "grad t (warp)"),
        ("Local energy", "grad log j"),
        ("Local energy", "sum grad log j"),
    ]
    ) for walker in walkers
]

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
    outfile="test.hdf5"
)


println("Energy: $(last(energies)) +- $(last(errors))")


#ws = get_weights("test.hdf5")
#
#flhf, flhf_warp = hellmann_feynman_force("test.hdf5")
#flpexact, flpexact_warp = pulay_force_exact("test.hdf5")
#flpvd, flpvd_warp = pulay_force_vd("test.hdf5")
#
#fhf = mean(flhf, Weights(ws))
#fhf_warp = mean(flhf_warp, Weights(ws))
#
#fpexact = mean(flpexact, Weights(ws))
#fpexact_warp = mean(flpexact_warp, Weights(ws))
#
#fpvd = mean(flpvd, Weights(ws))
#fpvd_warp = mean(flpvd_warp, Weights(ws))
#
#println("Force (exact):       $(fhf + fpexact)")
#println("Force (exact, warp): $(fhf_warp + fpexact_warp)")
#println("Force (vd):          $(fhf + fpvd)")
#println("Force (vd, warp):    $(fhf_warp + fpvd_warp)")
#
#pyplot()
#
#p1 = plot((flhf + flpvd)[2:end], reuse=false)
#plot!((flhf_warp + flpvd_warp)[2:end])
#display(p1)
#
p2 = plot(energies, ribbon=(errors, errors), fillalpha=0.2, reuse=false)
hline!([5], color="black")
hline!([pi^2/2], color="black")
display(p2)
