using Random
using LinearAlgebra


function diffuse_walker!(walker, ψ, τ, eref, sigma, model, rng::AbstractRNG)

    x = walker.configuration

    ψval = walker.ψstatus.value
    ∇ψ = walker.ψstatus.gradient
    #∇²ψ = walker.ψstatus_old.laplacian
    ∇²ψ = ψ.laplacian(x)
    v = cutoff_velocity(∇ψ/ψval, τ)

    gauss = √τ * randn(rng, Float64, size(x))
    drift = v*τ

    #x′ = x .+ v*τ .+ sqrt(τ)*randn(rng, Float64, size(x))
    x′ = x + drift + gauss

    ψval′ = ψ.value(x′)
    ∇ψ′ = ψ.gradient(x′)
    ∇²ψ′ = ψ.laplacian(x′)
    v′ = cutoff_velocity(∇ψ′ / ψval′, τ)
    new_drift = v′*τ

    #el′ = model.hamiltonian_recompute(ψ, x′) / ψval′
    
    forward = norm(gauss)^2
    backward = norm(gauss + drift + new_drift)^2

    t_prob = exp(1/2τ * (forward - backward))

#    num = exp.(-norm(x .- x′ .- v′*τ)^2 / 2τ)
#    denom = exp.(-norm(x′ .- x .- v*τ)^2 / 2τ)
    
    #acceptance = min(1.0, ψval′^2 / ψval^2 * num / denom)
    #acceptance = min(1.0, ψval′^2 / ψval^2 * t_prob)
    acceptance = 1.0

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

    el = model.hamiltonian(walker.ψstatus_old, x) / walker.ψstatus_old.value
    el′ = model.hamiltonian(walker.ψstatus, x) / walker.ψstatus.value

    s = eref - el
    s′ = eref - el′

    exponent = 0.5τ * (s + s′)
    #println(exponent)

    #println("Cache:     $el    $(walker.ψstatus_old.laplacian)    $(walker.ψstatus_old.value)")
    #println("Recompute: $el    $(ψ.laplacian(x))    $(ψ.value(x))")
    #println("$eref    $s    $s′    $el    $el′")

    mult = exp(0.5τ * (s + s′))
    walker.weight *= mult

end


function damp_timestep(el′, el, eref, sigma; stop=6, start=3)
    fbet = max(eref - el, eref - el′)
    clamp(1 - (fbet - start*sigma) / ((stop - start) * sigma), 0, 1)
end