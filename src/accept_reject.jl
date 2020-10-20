using Random
using LinearAlgebra
using Distributions


function move_walker!(walker, τ, ψ, rng::AbstractRNG)
    x = walker.configuration

    ψval = walker.ψstatus.value
    ∇ψ = walker.ψstatus.gradient
    ∇²ψ = walker.ψstatus.laplacian

    v = cutoff_velocity(∇ψ/ψval, τ)
    
    x′ = x .+ v*τ .+ sqrt(τ)*randn(rng, Float64, size(x))

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
    
    ratio = ψ′^2 / ψ^2

    v = cutoff_velocity(∇ψ/ψ, τ)
    v′ = cutoff_velocity(∇ψ′/ψ′, τ)

    num = exp.(-norm(x .- x′ .- 2v′*τ)^2 / 4τ)
    denom = exp.(-norm(x′ .- x .- 2v*τ)^2 / 4τ)

    p = min(1.0, ratio * num / denom)
     
    if sign(ψ) != sign(ψ′) || ψ′ == 0.0
        p = 0.0
    end

    return p
end

function accept_move!(walker, p, rng::AbstractRNG)
    # reject move if acceptance too small
    ξ = rand(rng)
    if p < ξ
        walker.configuration .= walker.configuration_old
        walker.ψstatus = deepcopy(walker.ψstatus_old)
    end
end