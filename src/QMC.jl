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

hamiltonian_sho(ψ, ψstatus, x) = -0.5*ψstatus.laplacian + 0.5*norm(x)^2*ψstatus.value
hamiltonian(ψ, ψstatus, x) = -0.5*ψstatus.laplacian

ψgauss(x) = exp(-norm(x)^2)


const a = 1.0
const α = a*cosh(1)
const β = a*sinh(1)

ψpib(x) = max(0, x[1].*(a .- x[1]))
#ψellipse(x) = max(0, 1 - (x[1]/α)^2 + (x[2]/β)^2)

ψtrial = WaveFunction(
    ψpib,
    x -> gradient_fd(ψpib, x),
    x -> laplacian_fd(ψpib, x)
)

model = Model(
    hamiltonian,
    ψtrial,
)

nwalkers = 400
num_blocks = 400
steps_per_block = 100
neq = 10

τ = 1e-2

rng = MersenneTwister(134)

walkers = generate_walkers(nwalkers, ψtrial, rng, Uniform(0., 1.), (1, 1))

#ψstatus = init_status(ψtrial, xstart)

observables = Dict(
    "Local energy" => (ψ, walker) -> hamiltonian(ψ, walker.ψstatus, walker.configuration) / walker.ψstatus.value,
    "ψ" => (ψ, walker) ->  walker.ψstatus.value,
    "∇ψ" => (ψ, walker) -> walker.ψstatus.gradient
)

accumulator = Accumulator(observables)

energies, errors = run_dmc!(model, walkers, τ, num_blocks, steps_per_block, 5.0; rng=rng, neq=neq)#; accumulator=accumulator)

print("$(last(energies)) +- $(last(errors))")

plot(energies, ribbon=(errors, errors), fillalpha=0.2)
hline!([5], color="black")
hline!([pi^2/2], color="black")