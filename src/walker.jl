using DataStructures

# Walkers carry a configuration and a weight, as well
# as the status of the wavefunction (its v,g,l on the current
# and previous configurations)
mutable struct Walker
    configuration::AbstractArray
    configuration_old::AbstractArray
    weight::Float64
    ψstatus::WaveFunctionStatus
    ψstatus_old::WaveFunctionStatus
end

# A fat walker is a walker along with a dictionary
# of data it carries, such as the local energy, or a
# history of certain observables for computing
# intgrals along its path.
# Covariances indicates which observables
# to compute covariances between, which
# requires products of observables.
mutable struct FatWalker
    walker::Walker
    observables::OrderedDict{String, Function}
    data::OrderedDict{String, CircularBuffer}
    covariances::Array{Tuple{String, String}}

    # including everything
    function FatWalker(walker, observables, data, covs)
        data["Weight"] = CircularBuffer(1)
        new(walker, observables, data, covs)
    end

    # no covariances
    function FatWalker(walker, observables, data) 
        FatWalker(walker, observables, data, [])
    end

    # no histories
    function  FatWalker(walker, observables, covs)
        data = OrderedDict()
        for (k, v) in observables
            data[k] = CircularBuffer(1)
        end
        data["Weight"] = CircularBuffer(1)
        new(walker, observables, data, covs)
    end

    # nothing
    FatWalker(walker, observables) = FatWalker(walker, observables, [])
end

function accumulate_observables!(fwalker, model, eref)
    for (key, func) in fwalker.observables
        val = func(fwalker, model, eref)
        push!(fwalker.data[key], val)
    end
    push!(fwalker.data["Weight"], fwalker.walker.weight)
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