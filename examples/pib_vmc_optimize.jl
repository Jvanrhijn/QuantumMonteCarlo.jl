using Distributions
using ProgressMeter
using DataStructures
using Random
using LinearAlgebra
using HDF5
using StatsBase
using Plots

using QuantumMonteCarlo

# Force computation settings and import
global a = 1.0
k = 1.0
da = 1e-5
    
# VMC settings
τ = 1e-1
nwalkers = 1
num_blocks = 1000
steps_per_block = 100
neq = num_blocks ÷ 10 
#neq = 1
#lag = trunc(Int64, steps_per_block)
#lag = steps_per_block
lag = 100
eref = 5.0/(2a)^2
niter = 40

hamiltonian(ψstatus, x) = -0.5*ψstatus.laplacian + 0.5k*a^2*ψstatus.value
hamiltonian_recompute(ψ, x) = -0.5*ψ.laplacian(x) + 0.5k*a^2*ψ.value(x)
hamiltonian′(ψstatus, x) = -0.5*ψstatus.laplacian + 0.5k*(a + da)^2*ψstatus.value
hamiltonian_recompute′(ψ, x) = -0.5*ψ.laplacian(x) + 0.5k*(a + da)^2*ψ.value(x)
include("forceutil_vmc.jl")
    
function optimize(warp)
    global a = 1.0

    as = [a]

    @showprogress for iter = 1:niter


        # Setting up the hamiltonian
        hamiltonian(ψstatus, x) = -0.5*ψstatus.laplacian + 0.5k*a^2*ψstatus.value
        hamiltonian_recompute(ψ, x) = -0.5*ψ.laplacian(x) + 0.5k*a^2*ψ.value(x)
        hamiltonian′(ψstatus, x) = -0.5*ψstatus.laplacian + 0.5k*(a + da)^2*ψstatus.value
        hamiltonian_recompute′(ψ, x) = -0.5*ψ.laplacian(x) + 0.5k*(a + da)^2*ψ.value(x)

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

        model = Model(
            hamiltonian,
            hamiltonian_recompute,
            ψtrial,
        )

        observables = OrderedDict(
            # Local energy
            "Local energy" => local_energy,
            # Gradients of local energy
            "grad el" => (fwalker, model, eref, xp) -> local_energy_gradient(fwalker, model, eref, xp, ψtrial′, τ; warp=warp),
            #"grad el (warp)" => (fwalker, model, eref, xp) -> local_energy_gradient(fwalker, model, eref, xp, ψtrial′, τ; warp=warp),
            # Gradients of log(ψ)
            "grad log psi" => (fwalker, model, eref, xp) -> log_psi_gradient(fwalker, model, eref, xp, ψtrial′, τ; warp=warp),
            #"grad log psi (warp)" => (fwalker, model, eref, xp) -> log_psi_gradient(fwalker, model, eref, xp, ψtrial′, τ; warp=warp),
            # Jacobians
            "grad log j" => (fwalker, model, eref, xp) -> jacobian_gradient_current(fwalker, model, eref, xp, ψtrial′, τ),
        )

        rng = MersenneTwister(16224267)

        # create "Fat" walkers
        walkers = QuantumMonteCarlo.generate_walkers(nwalkers, ψtrial, rng, Uniform(-a, a), 1)

        fat_walkers = [QuantumMonteCarlo.FatWalker(
            walker, 
            observables, 
            OrderedDict(
            ),
            [
                ("Local energy", o) for o in keys(observables)
            ]
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
            outfile="pib_vmc_opt.hdf5",
            verbosity=:quiet,
        );

        # compute forces
        force = h5open("pib_vmc_opt.hdf5", "r") do file
            elgrad = read(file, "grad el")[2:end]
            el = read(file, "Local energy")[2:end]
            energy = mean(el)
            logpsigrad = read(file, "grad log psi")[2:end]
            el_times_logpsigrad = read(file, "Local energy * grad log psi")[2:end]
            logjacgrad = read(file, "grad log j")[2:end]
            el_times_logjacgrad = read(file, "Local energy * grad log j")[2:end]

            -mean(elgrad 
                + (2*(el_times_logpsigrad - energy*logpsigrad) 
                + Float64(warp)*(el_times_logjacgrad - energy*logjacgrad)))

        end

        δt = 1e-1

        global a += force * δt
        append!(as, a)
    end
    return as
end

a_opt = (5/2k)^0.25

as = optimize(false)
as_warp = optimize(true)

plot(abs.(as .- a_opt), yscale=:log10, xlabel="Iteration", ylabel="|a - a*|", label="Bare")
plot!(abs.(as_warp .- a_opt), yscale=:log10, label="Warp")