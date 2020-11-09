using Random
using LinearAlgebra
using Distributions


function move_walker!(walker, τ, ψ, rng::AbstractRNG)
    x = walker.configuration

    ψval = walker.ψstatus.value
    ∇ψ = walker.ψstatus.gradient
    ∇²ψ = walker.ψstatus.laplacian

    v = ∇ψ / ψval
    
    # Move walker to x′ according to Langevin Itô diffusion
    # expression taken from https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm
    # with π(x) = ψ(x)², so ∇log(π) = 2∇log(ψ) = 2∇ψ/ψ.
    d = abs(ψval) / norm(∇ψ)
    Δx = v*τ + √τ * randn(rng, Float64, size(x))
    x′ = x + Δx

    # Update the walker with this new configuration
    walker.configuration_old .= deepcopy(x)
    walker.ψstatus_old = deepcopy(walker.ψstatus)

    walker.configuration .= x′
    walker.ψstatus.value = ψ.value(x′)
    walker.ψstatus.gradient .= ψ.gradient(x′)
    walker.ψstatus.laplacian = ψ.laplacian(x′)
end

function compute_acceptance!(walker, τ)
    x = walker.configuration_old
    x′ = walker.configuration
    ψ = walker.ψstatus_old.value
    ψ′ = walker.ψstatus.value

    ∇ψ = walker.ψstatus_old.gradient
    ∇ψ′ = walker.ψstatus.gradient
    
    # probability distribution ratio
    ratio = ψ′^2 / ψ^2

    v = ∇ψ / ψ
    v′ = ∇ψ′ / ψ′

    # MALA: compute the acceptance probability p
    # expression taken from https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm
    num = exp(-norm(x .- x′ .- v′ * τ)^2 / 2τ)
    denom = exp(-norm(x′ .- x .- v * τ)^2 / 2τ)

    p = min(1.0, ratio * num / denom)
     
    # Fixed-node: reject moves that change the sign of ψ
    if sign(ψ) != sign(ψ′)
        p = 0.0
    end

    return p
end

function accept_move!(walker, p, rng::AbstractRNG)
    # reject move with probability q = 1 - p
    # generate a uniform random variable on [0, 1]
    ξ = rand(rng, Uniform(0, 1))

    # if the acceptance probability < ξ, reject the proposed move
    if p < ξ
        # if rejected, restore the "old" walker configuration
        # and retrieve the old wave function values
        walker.configuration .= walker.configuration_old
        walker.ψstatus = deepcopy(walker.ψstatus_old)
    end
end