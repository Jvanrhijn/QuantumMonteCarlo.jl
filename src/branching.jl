function stochastic_reconfiguration!(walkers, rng::AbstractRNG)
    weights = map(w -> w.walker.weight, walkers)
    confs = [w.walker.configuration for w in walkers]
    new_walkers = sample(rng, walkers, Weights(weights), length(walkers); replace=true)
    new_walkers = [deepcopy(w) for w in new_walkers]
    
    for walker in new_walkers
        walker.walker.weight = 1.0
    end
    
    walkers .= new_walkers
end