using Random
using LinearAlgebra


function diffuse_walker!(walker, ψ, τ, rng::AbstractRNG)
    ∇ψ = walker.ψstatus.gradient
    ψval = walker.ψstatus.value
    v = cutoff_velocity(∇ψ/ψval, τ)

    x = walker.configuration
    
    x′ = x .+ v*τ .+ sqrt(τ)*randn(rng, Float64, size(x))
    
    ψval′ = ψ.value(x′)
    ∇ψ′ = ψ.gradient(x′)
    v′ = cutoff_velocity(∇ψ′ / ψval′, τ)
    
    num = exp.(-norm(x .- x′ .- v′*τ)^2/(2.0τ))
    denom = exp.(-norm(x′ .- x .- v*τ)^2/(2.0τ))
    
    acceptance = min(1.0, ψval′^2 / ψval^2 * num / denom)

    if ψval′ == 0.0 || sign(ψval′) != sign(ψval)
        acceptance = 0.0
    end
    
    if acceptance > rand(rng, Float64)
        walker.ψstatus_old.value = ψval
        walker.ψstatus_old.gradient .= ∇ψ
        walker.ψstatus_old.laplacian = walker.ψstatus.laplacian
        walker.configuration_old .= x

        walker.ψstatus.value = ψval′
        walker.ψstatus.gradient .= ∇ψ′
        walker.ψstatus.laplacian = ψ.laplacian(x′)
        walker.configuration .= x′
    end
end

function cutoff_velocity(v, τ)
    vnorm = norm(v)
    v * (-1 + sqrt(1 + 2*vnorm^2*τ))/(vnorm^2*τ)
end