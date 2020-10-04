function stochastic_reconfiguration!(walkers, rng::AbstractRNG)
    weights = map(w -> w.weight, walkers)
    confs = [walker.configuration for walker in walkers]
    new_walkers = sample(rng, walkers, Weights(weights), length(walkers); replace=true)
    new_walkers = [deepcopy(w) for w in new_walkers]
    
    for walker in new_walkers
        walker.weight = 1.0
    end
    
    walkers = new_walkers
end