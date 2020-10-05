include("wave_function_types.jl")
include("model.jl")
include("walker.jl")
include("accept_reject.jl")
include("branching.jl")
include("dmc.jl")
include("util.jl")
include("accumulator.jl")

using Distributions
using Plots


const a = 1.0
const da = 1e-5


ψpib(x) = max(0, x[1].*(a .- x[1]))
ψpib′(x) = max(0, x[1].*(a + da .- x[1]))

ψtrial = WaveFunction(
    ψpib,
    x -> gradient_fd(ψpib, x),
    x -> laplacian_fd(ψpib, x)
)

ψtrial′ = WaveFunction(
    ψpib′,
    x -> gradient_fd(ψpib′, x),
    x -> laplacian_fd(ψpib′, x)
)

hamiltonian(ψ, ψstatus, x) = -0.5*ψstatus.laplacian

model = Model(
    hamiltonian,
    ψtrial,
)

nwalkers = 1
num_blocks = 1000
steps_per_block = 100
neq = 10

τ = 1e-2

rng = MersenneTwister(134)

walkers = generate_walkers(nwalkers, ψtrial, rng, Uniform(0., 1.), (1, 1))

function elgrad(ψ, walker)
    el = hamiltonian(ψ, walker.ψstatus, walker.configuration) / walker.ψstatus.value
    el′ = -0.5*ψtrial′.laplacian(walker.configuration) / ψtrial′.value(walker.configuration)
    return (el′ - el) / da
end

function logpsigrad(ψ, walker)
    logψ = log(abs(walker.ψstatus.value))
    logψ′ = log(abs(ψtrial′.value(walker.configuration)))
    
    (logψ′ - logψ) / da
end

#mutable struct ForceData
#    ∇ₐel::Float64
#    ∇ₐlogψ::Float64
#    el∇ₐlogψ::Float64
#end

observables = Dict(
    "Local energy" => (ψ, walker) -> hamiltonian(ψ, walker.ψstatus, walker.configuration) / walker.ψstatus.value,
    "grad_a E_L" => elgrad,
    "grad_a log Psi" => logpsigrad,
    "E_L * grad_a log Psi" => (ψ, walker) -> hamiltonian(ψ, walker.ψstatus, walker.configuration) / walker.ψstatus.value * logpsigrad(ψ, walker),
    "grad_a log Jacobian" => (_, _) -> 0.0,
    "E_L * grad_a log Jacobian" => (_, _) -> 0.0,
    "grad_a E_L (warp)" => elgrad,
    "grad_a log Psi (warp)" => logpsigrad,
    "E_L * grad_a log Psi (warp)" => (ψ, walker) -> hamiltonian(ψ, walker.ψstatus, walker.configuration) / walker.ψstatus.value * logpsigrad(ψ, walker),
)

# IDEA: one field of accumulator should be a function
# that determines how observables are collected.
# The default will just be a loop over the dictionary of
# observable functions,
# for forces it will be more complicated
# First, let's do the HDF5 storage thing
accumulator = Accumulator(observables)

energies, errors = run_dmc!(model, walkers, τ, num_blocks, steps_per_block, 5.0; rng=rng, neq=neq, accumulator=accumulator, outfile="test.hdf5")

print("$(last(energies)) +- $(last(errors))")

plot(energies, ribbon=(errors, errors), fillalpha=0.2)
hline!([5], color="black")
hline!([pi^2/2], color="black")