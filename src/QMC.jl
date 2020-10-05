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

nwalkers = 100
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

function cutoff_tanh(d; a=0.1)
    b = a/5
    value = 0.5 * (1 + tanh((a - d)/b))
    deriv = -0.5 / b / cosh((a - d)/b)^2
    value, deriv
end

function node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′)
    d = abs(ψ) / norm(∇ψ)
    d′ = abs(ψ′) / norm(∇ψ′)
    n′ = ∇ψ′ / norm(∇ψ′)
    n = ∇ψ / norm(∇ψ)
    u, uderiv = cutoff_tanh(d)
    xwarp = x .+ (d - d′) * u * sign(ψ′) * n′
    jac = 1 - u + sign(ψ′*ψ) * dot(n, n′) * (u + (d - d′)*uderiv)
    xwarp, jac
end

function gradel(fwalker, model, eref)
    walker = fwalker.walker
    el = local_energy(fwalker, model, eref)
    el′ = -0.5 * ψtrial′.laplacian(walker.configuration) / ψtrial′.value(walker.configuration)
    (el′ - el) / da
end

function psi_sec(fwalker, model, eref)
    ψtrial′.value(fwalker.walker.configuration)
end

function gradpsi_sec(fwalker, model, eref)
    ψtrial′.gradient(fwalker.walker.configuration)
end

function psi_sec_prev(fwalker, model, eref)
    ψtrial′.value(fwalker.walker.configuration_old)
end

function gradpsi_sec_prev(fwalker, model, eref)
    ψtrial′.gradient(fwalker.walker.configuration_old)
end

function gradel_warp(fwalker, model, eref)
    walker = fwalker.walker
    ψ = walker.ψstatus.value
    ∇ψ = walker.ψstatus.gradient
    ψ′ = last(fwalker.data["ψ′"])
    ∇ψ′ = last(fwalker.data["∇ψ′"])
    x = walker.configuration
    xwarp, jac = node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′)

    el = -0.5 * walker.ψstatus.laplacian / walker.ψstatus.value
    el′ = -0.5 * ψtrial′.laplacian(xwarp) / ψtrial′.value(xwarp)

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

function grads_warp(fwalker, model, eref)
    walker = fwalker.walker

    ∇ₐel_warp = last(fwalker.data["grad el (warp)"])

    x = walker.configuration
    xprev = fwalker.walker.configuration_old

    ψ = walker.ψstatus.value
    ∇ψ = walker.ψstatus.gradient

    ψ′ = last(fwalker.data["ψ′"])
    ∇ψ′ = last(fwalker.data["∇ψ′"])

    ψprev = walker.ψstatus_old.value
    ∇ψprev = walker.ψstatus_old.gradient

    ψ′prev = last(fwalker.data["ψ′_old"])
    ∇ψ′prev = last(fwalker.data["∇ψ′_old"])

    # perform warp
    xwarp , _ = node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′)
    xwarpprev, _ = node_warp(xprev, ψprev, ∇ψprev, ψ′prev, ∇ψ′prev)

    el_prev = -0.5 * fwalker.walker.ψstatus_old.laplacian / fwalker.walker.ψstatus_old.value
    el_prev′ = -0.5 * ψtrial′.laplacian(xwarpprev) / ψtrial′.value(xwarpprev)

    ∇ₐel_prev_warp = (el_prev′ - el_prev) / da

    return -0.5 * (∇ₐel_warp + ∇ₐel_prev_warp) * τ
end

function gradt(fwalker, model, eref)
    ∇ψ = fwalker.walker.ψstatus_old.gradient
    ψ = fwalker.walker.ψstatus_old.value
    v = ∇ψ / ψ

    x′ = fwalker.walker.configuration
    x = fwalker.walker.configuration_old

    ∇ψsec = ψtrial′.gradient(x)
    ψsec = ψtrial′.value(x)
    vsec = ∇ψsec / ψsec

    t = -1/(2.0τ) * norm(x′ - x - v*τ)^2
    t′ = -1/(2.0τ) * norm(x′ - x - vsec*τ)^2

    return (t' - t) / da
end

function gradt(fwalker, model, eref)
    walker = fwalker.walker

    x = walker.configuration
    xprev = walker.configuration_old

    ψ = walker.ψstatus.value
    ∇ψ = walker.ψstatus.gradient
    v = ∇ψ / ψ

    ψ′ = last(fwalker.data["ψ′"])
    ∇ψ′ = last(fwalker.data["∇ψ′"])

    ψprev = walker.ψstatus_old.value
    ∇ψprev = walker.ψstatus_old.gradient

    ψ′prev = last(fwalker.data["ψ′_old"])
    ∇ψ′prev = last(fwalker.data["∇ψ′_old"])

    # perform warp
    xwarp , _ = node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′)
    xwarpprev, _ = node_warp(xprev, ψprev, ∇ψprev, ψ′prev, ∇ψ′prev)

    # compute warped drift
    ∇ψ′_old_warp = ψtrial′.gradient(xwarpprev)
    ψ′_old_warp = ψtrial′.value(xwarpprev)
    vsec_warp = ∇ψ′_old_warp / ψ′_old_warp

    t = -1/(2.0τ) * norm(x - xprev - v*τ)^2
    t′ = -1/(2.0τ) * norm(xwarp - xwarpprev - vsec_warp*τ)^2

    return (t' - t) / da
end

observables = OrderedDict(
    "ψ′" => psi_sec,
    "∇ψ′" => gradpsi_sec,
    "ψ′_old" => psi_sec,
    "∇ψ′_old" => gradpsi_sec,
    "Local energy" => local_energy,
    "grad el" => gradel,
    "grad el (warp)" => gradel_warp,
    "grad log psi" => (fwalker, model, _) -> (log(abs(ψtrial′.value(fwalker.walker.configuration))) - log(abs(ψtrial.value(fwalker.walker.configuration)))) / da,
    "grad s" => grads,
    "grad t" => gradt,
    "grad s (warp)" => grads_warp,
    "grad t (warp)" => gradt,
)

walkers = generate_walkers(nwalkers, ψtrial, rng, Uniform(0., 1.), (1, 1))

fat_walkers = [FatWalker(
    walker, 
    observables, 
    OrderedDict(
        "ψ′" => CircularBuffer(1),
        "∇ψ′" => CircularBuffer(1),
        "ψ′_old" => CircularBuffer(1),
        "∇ψ′_old" => CircularBuffer(1),
        "Local energy" => CircularBuffer(1),
        "grad el" => CircularBuffer(1),
        "grad el (warp)" => CircularBuffer(1),
        "grad log psi" => CircularBuffer(1),
        "grad s" => CircularBuffer(steps_per_block),
        "grad t" => CircularBuffer(steps_per_block),
        "grad s (warp)" => CircularBuffer(steps_per_block),
        "grad t (warp)" => CircularBuffer(steps_per_block),
    ),
    [
        ("Local energy", "grad log psi"),
        ("Local energy", "grad s"),
        ("Local energy", "grad t"),
        ("Local energy", "grad s (warp)"),
        ("Local energy", "grad t (warp)"),
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

fvd, fexact = h5open("test.hdf5", "r") do file
    els = read(file, "Local energy")
    ws = read(file, "Weight")

    ∇ₐel = read(file, "grad el (warp)")

    ∇ₐlogψ = read(file, "grad log psi")
    el∇ₐlogψ = read(file, "Local energy * grad log psi")

    ∇ₐs = read(file, "grad s (warp)")
    el∇ₐs = read(file, "Local energy * grad s (warp)")

    ∇ₐt = read(file, "grad t (warp)")
    el∇ₐt = read(file, "Local energy * grad t (warp)")

    energy = mean(els, Weights(ws))

    fvd = mean(-(∇ₐel .+ 2.0(el∇ₐlogψ .- energy*∇ₐlogψ) .+ el∇ₐs .- energy*∇ₐs), Weights(ws))
    fexact = mean(-(∇ₐel .+ el∇ₐt - energy*∇ₐt .+ el∇ₐs .- energy*∇ₐs), Weights(ws))

    return fvd, fexact
end

println("VD force: $fvd")
println("Exact force: $fexact")

plot(energies, ribbon=(errors, errors), fillalpha=0.2)
hline!([5], color="black")
hline!([pi^2/2], color="black")
