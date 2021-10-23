# Accumulator holds a hashmap of key-function pairs,
# which allow one to compute in principle arbitrary
# observables during a DMC run.
using DataStructures
using StatsBase
using HDF5


mutable struct Accumulator
    block_data::OrderedDict
    block_averages::OrderedDict{String, Float64}

    function Accumulator(fat_walkers)  
        # obtain observables from walkers
        ks = collect(keys(fat_walkers[1].data))

        # covariances
        for (k1, k2) in fat_walkers[1].covariances
            k = k1 * " * " * k2
            push!(ks, k)
        end

        block_data = OrderedDict(k => [] for k in ks)
        block_averages = OrderedDict{String, Float64}(k => 0.0 for k in ks)

        block_data["Weight"] = []
        block_averages["Weight"] = 0.0

        new(block_data, block_averages)
    end

end

function average_ensemble!(fat_walkers, accumulator)
    # TODO: Deal with covariance observables

    weights = map(w -> last(w.walker.weight), fat_walkers)
    total_weight = sum(weights)

    # obtain observables from walkers
    # TODO: make FatWalker generation more foolproof;
    # don't allow specification of different sets of
    # observables per walker, each walker should be a clone
    # of the others
    ks = keys(accumulator.block_data)
    data = OrderedDict(key => [] for key in ks)

    for fwalker in fat_walkers
        for key in fwalker.observable_names
            # sum over the walker's history
            history_sum = fwalker.history_sums[key]
            #push!(data[key], homsum(fwalker.data[key].buffer))
            push!(data[key], history_sum)
        end
    end

    # deal with covariance observables
    for fwalker in fat_walkers
        for (k1, k2) in fwalker.covariances
            k = k1 * " * " * k2
            histsum1 = fwalker.history_sums[k1]
            histsum2 = fwalker.history_sums[k2]
            push!(data[k], histsum1*histsum2)
        end
    end

    # average over ensemble data
    for (key, value) in data
        if key == "Weight"
            continue
        end
        push!(accumulator.block_data[key], mean(value, Weights(weights)))
    end

    push!(accumulator.block_data["Weight"], total_weight)
end

function average_block!(accumulator)
    weights = Float64.(accumulator.block_data["Weight"])
    total_weight = sum(weights)

    for (key, value) in accumulator.block_data
        accumulator.block_averages[key] = mean(value, Weights(weights))
    end
    
    accumulator.block_averages["Weight"] = total_weight

    ks = keys(accumulator.block_data)
    accumulator.block_data = OrderedDict(k => [] for k in ks)
end

function write_to_file!(accumulator, file)
    g = HDF5.root(file)
    for (key, value) in accumulator.block_averages
        # if dataset doesn't exist yet, create it
        if !exists(g, key)
            d_create(g, key, Float64, ((1, size(value)...), (-1, size(value)...)), "chunk", (1,size(value)...))
        end
        dset = d_open(g, key)
        dim = size(dset)
        new_dim = (dim[1]+1, dim[2:end]...)
        set_dims!(dset, new_dim)
        dset[end] = value
    end

    ks = keys(accumulator.block_averages)
    accumulator.block_averages = OrderedDict{String, Float64}(k => 0.0 for k in ks)
end

function reset_collection!(collection)
    ks = keys(collection)
    OrderedDict(k => [] for k in ks)
end