# Accumulator holds a hashmap of key-function pairs,
# which allow one to compute in principle arbitrary
# observables during a DMC run.
using DataStructures
using StatsBase
using HDF5


mutable struct Accumulator
    observables::Dict{String, Function}
    ensemble_data::Dict
    block_data::Dict
    block_averages::Dict
    accumulate_observables!::Function

    function Accumulator(observables, accumulate_observables!)  
        ensemble_data = Dict()
        block_data = Dict()
        block_averages = Dict()

        for (key, value) in observables
            if key == "Weight"
                throw(ArgumentError("Weight not allowed as an observable name"))
            end
            ensemble_data[key] = []
            block_data[key] = []
            block_averages[key] = []
        end
        ensemble_data["Weight"] = []
        block_data["Weight"] = []
        block_averages["Weight"] = []

        new(observables, ensemble_data, block_data, block_averages, accumulate_observables!)
    end

    Accumulator(observables) = Accumulator(observables, accumulate_observables_default!)

end

function accumulate_observables_default!(walker, model, accumulator)
    for (key, value) in accumulator.observables
        val = value(model.wave_function, walker)
        push!(accumulator.ensemble_data[key], val)
    end
    weights = accumulator.ensemble_data["Weight"]
    push!(weights, walker.weight)
end

function average_ensemble!(accumulator)
    weights = Float64.(accumulator.ensemble_data["Weight"])
    total_weight = sum(weights)

    for (key, value) in accumulator.ensemble_data
        accumulator.ensemble_data[key] = mean(value, Weights(weights))
    end

    for (key, value) in accumulator.ensemble_data
        push!(accumulator.block_data[key], value)
    end

    accumulator.block_data["Weight"][end] = total_weight

    accumulator.ensemble_data = Dict()
    for (key, _) in accumulator.observables
        accumulator.ensemble_data[key] = []
    end
    accumulator.ensemble_data["Weight"] = []

end

function average_block!(accumulator)
    weights = Float64.(accumulator.block_data["Weight"])
    total_weight = sum(weights)

    for (key, value) in accumulator.block_data
        accumulator.block_data[key] = mean(value, Weights(weights))
    end
    
    for (key, value) in accumulator.block_data
        push!(accumulator.block_averages[key], value)
    end

    accumulator.block_averages["Weight"][end] = total_weight

    accumulator.block_data = Dict()
    for (key, _) in accumulator.observables
        accumulator.block_data[key] = []
    end
    accumulator.block_data["Weight"] = []

end

function write_to_file!(accumulator, file)
    g = root(file)
    for (key, value) in accumulator.block_averages
        # if dataset doesn't exist yet, create it
        if !exists(g, key)
            d_create(g, key, Float64, ((1, size(value)...), (-1, size(value)...)), "chunk", (1,size(value)...))
        end
        dset = d_open(g, key)
        dim = size(dset)
        new_dim = (dim[1]+1, dim[2:end]...)
        set_dims!(dset, new_dim)
        dset[end, :] = value[:]
    end

    accumulator.block_averages = Dict()
    for (key, _) in accumulator.observables
        accumulator.block_averages[key] = []
    end
    accumulator.block_averages["Weight"] = []

end