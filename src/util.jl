function gradient_fd(f, x; dx=1e-5)
    dim = length(x)
    out = zeros(dim)
    for i = 1:dim
        dxx = zeros(dim)
        dxx[i] = dx
        out[i] = (f(x .+ dxx) - f(x .- dxx)) ./ (2.0*dx)
    end
    out
end

function laplacian_fd(f, x; dx=1e-5)
    dim = length(x)
    out = 0.0
    val = f(x)
    for i = 1:dim
        dxx = zeros(dim)
        dxx[i] = dx
        out += (f(x .+ dxx) - 2.0val + f(x .- dxx)) / dx^2
    end
    out
end

function homsum(array::Array{Any})
    converted = reshape([array...], size(array)...)
    sum(converted)
end


num_digits = Int32 ∘ ceil ∘ log10


function cutoff_velocity(v, τ; a=0.5)
    vnorm = norm(v)
    return vnorm > 0 ? v * (-1 + sqrt(1 + 2*a*vnorm^2*τ))/(a*vnorm^2*τ) : 0.0
end