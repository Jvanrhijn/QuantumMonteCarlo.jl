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

nwalkers = 25
num_blocks = 100
steps_per_block = 100
neq = 10

τ = 1e-2

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

rng = MersenneTwister(134)

local_energy(fwalker, model, eref) = hamiltonian(model.wave_function, fwalker.walker.ψstatus, fwalker.walker.configuration) / fwalker.walker.ψstatus.value

function gradel(fwalker, model, eref)
    walker = fwalker.walker
    el = local_energy(fwalker, model, eref)
    el′ = -0.5 * ψtrial′.laplacian(walker.configuration) / ψtrial′.value(walker.configuration)
    (el′ - el) / da
end

function grads(fwalker, model, eref)
    ∇ₐel = last(fwalker.data["grad el"])
    xprev = fwalker.walker.configuration_old
    el_prev = -0.5 * fwalker.walker.ψstatus_old.laplacian / fwalker.walker.ψstatus_old.value
    el_prev′ = -0.5 * ψtrial′.laplacian(xprev) / ψtrial′.value(xprev)
    ∇ₐel_prev = (el_prev′ - el_prev) / da
    return -0.5 * (∇ₐel + ∇ₐel_prev) * τ
end

observables = OrderedDict(
    "Local energy" => local_energy,
    "grad el" => gradel,
    "grad log psi" => (fwalker, model, _) -> (log(abs(ψtrial′.value(fwalker.walker.configuration))) - log(abs(ψtrial.value(fwalker.walker.configuration)))) / da,
    "grad s" => grads,
)

walkers = generate_walkers(nwalkers, ψtrial, rng, Uniform(0., 1.), (1, 1))

fat_walkers = [FatWalker(
    walker, 
    observables, 
    OrderedDict(
        "Local energy" => CircularBuffer(1),
        "grad el" => CircularBuffer(1),
        "grad log psi" => CircularBuffer(1),
        "grad s" => CircularBuffer(steps_per_block),
    ),
    [
        ("Local energy", "grad log psi"),
        ("Local energy", "grad s")
    ]
    ) for walker in walkers]

energies, errors = run_dmc!(
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

f = h5open("test.hdf5", "r") do file
    els = read(file, "Local energy")
    ws = read(file, "Weight")
    ∇ₐel = read(file, "grad el")
    ∇ₐlogψ = read(file, "grad log psi")
    el∇ₐlogψ = read(file, "Local energy * grad log psi")
    ∇ₐs = read(file, "grad s")
    el∇ₐs = read(file, "Local energy * grad s")
    energy = mean(els, Weights(ws))
    mean(-(∇ₐel .+ 2.0(el∇ₐlogψ .- energy*∇ₐlogψ) .+ el∇ₐs .- energy*∇ₐs), Weights(ws))
end

println("VD force: $f")

plot(energies, ribbon=(errors, errors), fillalpha=0.2)
hline!([5], color="black")
hline!([pi^2/2], color="black")
