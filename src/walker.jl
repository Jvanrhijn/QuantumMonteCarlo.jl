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
    square_displacement::Float64
    square_displacement_times_acceptance::Float64

    Walker(configuration, ψstatus) = new(configuration, deepcopy(configuration), 1.0, ψstatus, deepcopy(ψstatus), 0.0, 0.0)
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
    observable_names::Array{String}
    history_sums::OrderedDict{String, Any}

    # including everything
    function FatWalker(walker, observables, data, covs)
        data["Weight"] = CircularBuffer(1)
        data_keys = keys(data)
        observable_names = collect(keys(observables))
        history_sums = OrderedDict{String, Any}(k => 0.0 for k in observable_names)
        # only need to add buffers that are not
        # already given
        for (key, value) in observables
            if !(key in data_keys)
                data[key] = CircularBuffer(1)
            end
        end
        new(walker, observables, data, covs, observable_names, history_sums)
    end

    # no covariances
    function FatWalker(walker, observables, data::Dict{String, CircularBuffer}) 
        FatWalker(walker, observables, data, [])
    end

    # no histories
    function  FatWalker(walker, observables, covs::Array{Tuple{String, String}})
        data = OrderedDict()
        for (k, v) in observables
            data[k] = CircularBuffer(1)
        end
        data["Weight"] = CircularBuffer(1)
        observable_names = collect(keys(observables))
        history_sums = OrderedDict(k => 0.0 for k in observable_names)
        new(walker, observables, data, covs, observable_names, history_sums)
    end

    # only observables
    FatWalker(walker, observables) = FatWalker(walker, observables, Tuple{String, String}[])

    # nothing
    FatWalker(walker) = FatWalker(walker, OrderedDict{String, Function}())
end

function accumulate_observables!(fwalker, model, eref)
    #for (key, func) in fwalker.observables
    for key in fwalker.observable_names
        func = fwalker.observables[key]
        new_val = func(fwalker, model, eref)
        
        # if the history is saturated, subtract the oldest from sum
        if length(fwalker.data[key]) == fwalker.data[key].capacity
            fwalker.history_sums[key] = fwalker.history_sums[key] .- first(fwalker.data[key])
        end

        push!(fwalker.data[key], new_val)

        # add new value to history sum
        fwalker.history_sums[key] = fwalker.history_sums[key] .+ new_val

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
        walkers[i] = Walker(conf, ψstatus)
    end
    walkers
end