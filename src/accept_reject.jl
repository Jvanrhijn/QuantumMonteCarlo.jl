using Random
using LinearAlgebra


function diffuse_walker!(walker, ψ, τ, eref, model, rng::AbstractRNG)

    ψval = walker.ψstatus.value
    ∇ψ = walker.ψstatus.gradient
    ∇²ψ = walker.ψstatus_old.laplacian
    v = cutoff_velocity(∇ψ/ψval, τ)

    x = walker.configuration
    el = model.hamiltonian(walker.ψstatus, x) / ψval

    x′ = x .+ v*τ .+ sqrt(τ)*randn(rng, Float64, size(x))

    ψval′ = ψ.value(x′)
    ∇ψ′ = ψ.gradient(x′)
    ∇²ψ′ = ψ.laplacian(x′)
    v′ = cutoff_velocity(∇ψ′ / ψval′, τ)

    el′ = model.hamiltonian_recompute(ψ, x′) / ψval′
    
    num = exp.(-norm(x .- x′ .- v′*τ)^2/(2.0τ))
    denom = exp.(-norm(x′ .- x .- v*τ)^2/(2.0τ))
    
    acceptance = min(1.0, ψval′^2 / ψval^2 * num / denom)

    if ψval′ == 0.0 || sign(ψval′) != sign(ψval)
        acceptance = 0.0
    end

    walker.ψstatus_old.value = ψval
    walker.ψstatus_old.gradient = ∇ψ
    walker.ψstatus_old.laplacian = ∇²ψ
    walker.configuration_old = x

    if acceptance > rand(rng, Float64)
        walker.ψstatus.value = ψval′
        walker.ψstatus.gradient = ∇ψ′
        walker.ψstatus.laplacian = ∇²ψ′
        walker.configuration = x′
    end

    # update displacement
    dx = walker.configuration - walker.configuration_old
    walker.square_displacement += norm(dx)^2
    walker.square_displacement_times_acceptance += acceptance*norm(dx)^2

    # compute effective time step
    if walker.square_displacement > 0.0
        τₑ = τ * walker.square_displacement_times_acceptance / walker.square_displacement
    else
        τₑ = τ
    end

    # umrigar's branching factor cutoff
    v = walker.ψstatus.gradient / walker.ψstatus.value
    vold = walker.ψstatus_old.gradient / walker.ψstatus_old.value
    ratio = norm(cutoff_velocity(v, τₑ)) / norm(v)
    ratio_old = norm(cutoff_velocity(vold, τₑ)) / norm(vold)

    s = (eref - el) #* ratio_old
    s′ = (eref - el′) #* ratio

    # update walker weight
    p = acceptance
    q = 1 - p
    if p != 0.0
        walker.weight *= exp((0.5 * p * (s + s′) + q*s) * τₑ)
    else
        walker.weight *= exp(q*s*τₑ)
    end
    #println(walker.weight)
    #walker.weight *= exp(0.5 * (s + s′) * τₑ)

end
