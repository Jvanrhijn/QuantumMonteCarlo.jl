# Accumulator holds a hashmap of key-function pairs,
# which allow one to compute in principle arbitrary
# observables during a DMC run.
using DataStructures

mutable struct Accumulator
    observables::Dict{String, Function}
    ensemble_data::DefaultDict{String, Array}
    Accumulator(observables) = new(observables, DefaultDict([]))
end

function accumulate_observables!(walker, model, accumulator)
    for (key, value) in accumulator.observables
        val = value(model.wave_function, walker)
        accumulator.ensemble_data[key].push(val)
    end
    accumulator.ensemble_data["Weight"].push(walker.weight)
end

function average_ensemble!(accumulator)
    weights = accumulator.ensemble_data["Weight"]
    total_weight = sum(weights)
    for (key, value) in accumulator.ensemble_data
        accumulator.ensemble_data[key] = mean(value, Weights(weights))
    end
    accumulator.ensemble_data["Weight"] = total_weight
end

function average_block!(accumulator)
    
end