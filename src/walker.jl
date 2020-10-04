mutable struct Walker
    configuration::AbstractArray
    configuration_old::AbstractArray
    weight::Float64
    ψstatus::WaveFunctionStatus
    ψstatus_old::WaveFunctionStatus
end

function generate_walkers(nwalker, ψ, rng, distribution, dimension)
    walkers = Vector{Walker}(undef, nwalker)
    for i = 1:nwalker
        conf = rand(rng, distribution, dimension)
        ψval = ψ.value(conf)
        ∇ψ = ψ.gradient(conf)
        ∇²ψ = ψ.laplacian(conf)
        ψstatus = WaveFunctionStatus(ψval, ∇ψ, ∇²ψ)
        walkers[i] = Walker(conf, conf, 1.0, ψstatus, ψstatus)
    end
    walkers
end