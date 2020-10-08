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