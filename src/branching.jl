function stochastic_reconfiguration!(walkers, rng::AbstractRNG)
    weights = map(w -> w.walker.weight, walkers)
    global_weight = mean(weights)
    confs = [w.walker.configuration for w in walkers]
    new_walkers = sample(rng, walkers, Weights(weights), length(walkers); replace=true)
    new_walkers = [deepcopy(w) for w in new_walkers]
    
    for walker in new_walkers
        walker.walker.weight = global_weight
    end
    
    walkers .= new_walkers
end

function simple_branching!(walkers, rng::AbstractRNG)
    max_copies = 50
    to_delete = []
    copies = []
    #weights = map(w.walker.weight, walkers)
    for (i, fwalker) in enumerate(walkers)
        walker = fwalker.walker
        if walker.weight > 1.0
            num_copies = trunc(Int64, walker.weight + rand(rng)) - 1
            println(num_copies)
            if num_copies > max_copies
                throw(DomainError("Too many walker copies made"))
            end
            push!(copies, deepcopy(fwalker))
        elseif walker.weight < rand(rng)
            push!(to_delete, i)
        end
    end
    # delete walkers-to-delete
    deleteat!(walkers, to_delete)
    # append copies
    append!(walkers, copies)
end