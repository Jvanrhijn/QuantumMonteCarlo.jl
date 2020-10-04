using Random


function diffuse_walker!(walker, ψ, rng::AbstractRNG)
    ∇ψ = walker.ψstatus.gradient
    ψval = walker.ψstatus.value
    drift = ∇ψ/ψval

    x = walker.configuration
    
    x′ = x .+ drift*τ .+ sqrt(τ)*randn(rng, Float64, size(x))
    
    ψval′ = ψ.value(x′)
    ∇ψ′ = ψ.gradient(x′)
    drift′ = ∇ψ′ / ψval′
    
    denom = exp.(-norm(x′ .- x .- drift*τ)^2/(2.0τ))
    num = exp.(-norm(x .- x′ .- drift′*τ)^2/(2.0τ))
    
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