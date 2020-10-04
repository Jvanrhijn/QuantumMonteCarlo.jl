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

nwalkers = 100
num_blocks = 400
steps_per_block = 100
neq = 10

τ = .5e-2

rng = MersenneTwister(134)

walkers = generate_walkers(nwalkers, ψtrial, rng, Uniform(0., 1.), (1, 1))

function elgrad(ψ, walker)
    el = hamiltonian(ψ, walker.ψstatus, walker.configuration) / walker.ψstatus.value
    el′ = -0.5*ψtrial′.laplacian(walker.configuration) / ψtrial′.value(walker.configuration)
    return (el′ - el) / da
end

observables = Dict(
    "Local energy" => (ψ, walker) -> hamiltonian(ψ, walker.ψstatus, walker.configuration) / walker.ψstatus.value,
    "ψ" => (ψ, walker) ->  walker.ψstatus.value,
    "∇ψ" => (ψ, walker) -> walker.ψstatus.gradient,
    "∇ₐel" => elgrad
)

accumulator = Accumulator(observables)

energies, errors = run_dmc!(model, walkers, τ, num_blocks, steps_per_block, 5.0; rng=rng, neq=neq, accumulator=accumulator)

print("$(last(energies)) +- $(last(errors))")

plot(energies, ribbon=(errors, errors), fillalpha=0.2)
hline!([5], color="black")
hline!([pi^2/2], color="black")