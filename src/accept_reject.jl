using Random


# TODO: make this an Iterator?
function accept_reject_diffuse!(ψ, τ, ψ_status, x, rng::AbstractRNG)
    ∇ψ = ψ_status.gradient
    ψval = ψ_status.value
    drift = ∇ψ/ψval
    
    xprop = x .+ drift*τ .+ sqrt(τ)*randn(rng, Float64, size(x))
    
    ψval_prop = ψ.value(xprop)
    ∇ψ_prop = ψ.gradient(xprop)
    drift_prop = ∇ψ_prop / ψval_prop
    
    denom = exp.(-norm(xprop .- x .- drift*τ)^2/(2.0τ))
    num = exp.(-norm(x .- xprop .- drift_prop*τ)^2/(2.0τ))
    
    acceptance = min(1.0, ψval_prop^2 / ψval^2 * num / denom)
    
    if acceptance > rand(rng, Float64)
        ψ_status.value = ψval_prop
        ψ_status.gradient = ∇ψ_prop
        ψ_status.laplacian = ψ.laplacian(xprop)
        x .= xprop
    end
end