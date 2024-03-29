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

# Setting up the hamiltonian
hamiltonian(ψstatus, x) = -0.5*ψstatus.laplacian
hamiltonian_recompute(ψ, x) = -0.5*ψ.laplacian(x)
hamiltonian′(ψstatus, x) = -0.5*ψstatus.laplacian
hamiltonian_recompute′(ψ, x) = -0.5*ψ.laplacian(x)

include("forceutil_vmc.jl")

# VMC settings
τ = 10e-2
nwalkers = 1
num_blocks = 1000
steps_per_block = 10
#neq = num_blocks ÷ 10 
neq = 1
#lag = trunc(Int64, steps_per_block)
#lag = steps_per_block
lag = 100
eref = 5.0/(2a)^2

# Trial wave function
function ψpib(x::AbstractArray)
    a^2 - x[1]^2
end

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

warp_factors = [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
approximate_warp_functions = [
			      [
			       "grad el (warp) ($wf)" => (fwalker, model, eref, xp) -> local_energy_gradient(fwalker, model, eref, xp, ψtrial′, τ; warp=true, warpfac=wf),
			       "grad el pathak ($wf)" => (fwalker, model, eref, xp) -> local_energy_gradient_pathak(fwalker, model, eref, xp, ψtrial′, τ; ϵ=wf),
			       "grad log psi (warp) ($wf)" => (fwalker, model, eref, xp) -> log_psi_gradient(fwalker, model, eref, xp, ψtrial′, τ; warp=true, warpfac=wf),
			       "grad log j ($wf)" => (fwalker, model, eref, xp) -> jacobian_gradient_current(fwalker, model, eref, xp, ψtrial′, τ, warpfac=wf),
			       "grad log j approx ($wf)" => (fwalker, model, eref, xp) -> jacobian_gradient_current_approx(fwalker, model, eref, xp, ψtrial′, τ, warpfac=wf),
			       "grad log psi pathak ($wf)" => (fwalker, model, eref, xp) -> log_psi_gradient_pathak(fwalker, model, eref, xp, ψtrial′, τ; ϵ=0.1wf),
			       ] for wf in warp_factors
]

approximate_warp_functions = OrderedDict(vcat(approximate_warp_functions...))

observables = OrderedDict(
    # Local energy
    "Local energy" => local_energy,
    "Local energy new" => local_energy_new,
    # Gradients of local energy
    "grad el" => (fwalker, model, eref, xp) -> local_energy_gradient(fwalker, model, eref, xp, ψtrial′, τ; warp=false),
    "grad el (warp)" => (fwalker, model, eref, xp) -> local_energy_gradient(fwalker, model, eref, xp, ψtrial′, τ; warp=true),
    # Gradients of log(ψ)
    "grad log psi" => (fwalker, model, eref, xp) -> log_psi_gradient(fwalker, model, eref, xp, ψtrial′, τ; warp=false),
    "grad log psi (warp)" => (fwalker, model, eref, xp) -> log_psi_gradient(fwalker, model, eref, xp, ψtrial′, τ; warp=true),
    # Jacobians
    "grad log j" => (fwalker, model, eref, xp) -> jacobian_gradient_current(fwalker, model, eref, xp, ψtrial′, τ),
    "sum grad log j" => (fwalker, model, eref, xp) -> jacobian_gradient_previous(fwalker, model, eref, xp, ψtrial′, τ),
    "grad log j approx" => (fwalker, model, eref, xp) -> jacobian_gradient_current_approx(fwalker, model, eref, xp, ψtrial′, τ),
    "sum grad log j approx" => (fwalker, model, eref, xp) -> jacobian_gradient_previous_approx(fwalker, model, eref, xp, ψtrial′, τ),
    # S, T with and without warp
    "grad g" => (fwalker, model, eref, xp) -> greens_function_gradient(fwalker, model, eref, xp, ψtrial′, τ; usepq=false, warp=false),
    "grad g (warp)" => (fwalker, model, eref, xp) -> greens_function_gradient(fwalker, model, eref, xp, ψtrial′, τ; usepq=false, warp=true),
    # S, T with p and q derivatives
    "grad g (p/q)" => (fwalker, model, eref, xp) -> greens_function_gradient(fwalker, model, eref, xp, ψtrial′, τ; usepq=true, warp=false),
    "grad g (warp, p/q)" => (fwalker, model, eref, xp) -> greens_function_gradient(fwalker, model, eref, xp, ψtrial′, τ; usepq=true, warp=true),
)

observables = merge(observables, approximate_warp_functions)

rng = MersenneTwister(16224267)

# create "Fat" walkers
walkers = QuantumMonteCarlo.generate_walkers(nwalkers, ψtrial, rng, Uniform(-0.1, 0.1), 1)

fat_walkers = [QuantumMonteCarlo.FatWalker(
    walker, 
    observables, 
    OrderedDict(
        "grad g" => CircularBuffer(lag),
        "grad g (warp)" => CircularBuffer(lag),
        "grad g (p/q)" => CircularBuffer(lag),
        "grad g (warp, p/q)" => CircularBuffer(lag),
        "sum grad log j" => CircularBuffer(lag),
        "sum grad log j approx" => CircularBuffer(lag),
    ),
    vcat(
        [
            ("Local energy", o) for o in keys(observables)
        ], 
        [
            ("Local energy new", o) for o in keys(observables)
        ] 
    )
    ) for walker in walkers
]

#fat_walkers = [QuantumMonteCarlo.FatWalker(walker) for walker in walkers]

energies, errors = QuantumMonteCarlo.run_vmc!(
    model, 
    fat_walkers, 
    τ, 
    num_blocks, 
    steps_per_block, 
    rng=rng, 
    neq=neq, 
    outfile="pib_vmc.hdf5",
    verbosity=:loud,
);