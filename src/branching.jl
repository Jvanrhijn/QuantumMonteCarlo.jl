function stochastic_reconfiguration!(walkers, rng::AbstractRNG)
    weights = map(w -> w.walker.weight, walkers)
    global_weight = mean(weights)
    confs = [w.walker.configuration for w in walkers]
    new_walkers = sample(rng, walkers, Weights(weights), length(walkers); replace=true)
    new_walkers = [deepcopy(w) for w in new_walkers]
    
    for walker in new_walkers
        walker.walker.weight = global_weight
        #walker.walker.weight = 1.0
    end
    
    walkers .= new_walkers
end

function stochastic_reconfiguration_pyqmc!(walkers, rng::AbstractRNG)
    weights = map(w -> w.walker.weight, walkers)
    nconf = length(weights)
    global_weight = mean(weights)
    wtot = sum(weights)
    probability = cumsum(weights / wtot)
    base = rand(rng)
    newinds = searchsortedfirst.(Ref(probability), (base .+ collect(1:nconf) / nconf) .% 1.0)

    new_walkers = walkers[newinds]

    for w in new_walkers
        w.walker.weight = global_weight
    end

    walkers .= [deepcopy(w) for w in new_walkers]
end

function optimal_stochastic_reconfiguration!(walkers, rng::AbstractRNG)
    weights = map(w -> w.walker.weight, walkers)
    global_weight = mean(weights)

    positive_walkers = filter(w -> w.walker.weight / global_weight  >= 1, walkers)
    negative_walkers = filter(w -> w.walker.weight / global_weight < 1, walkers)
    npos = length(positive_walkers)
    nneg = length(negative_walkers)

    # compute number of reconfigurations to perform
    positive_weights = map(w -> w.walker.weight, positive_walkers)
    negative_weights = map(w -> w.walker.weight, negative_walkers)

    nreconf = trunc(Int64, sum(abs.(positive_weights ./ global_weight .- 1)) + rand(rng))

    nreconf = min(npos, nneg, nreconf)

    # destroy nreconf negative walkers
    to_destroy = sample(rng, collect(1:nneg), nreconf; replace=false)
    deleteat!(negative_walkers, sort(to_destroy))

    # duplicate nreconf positive walkers
    to_duplicate = sample(rng, collect(1:npos), nreconf; replace=false)
    duplicates = [deepcopy(w) for w in positive_walkers[to_duplicate]]
    append!(positive_walkers, duplicates)

    walkers .= vcat(positive_walkers, negative_walkers)

    for w in walkers
        w.walker.weight = global_weight
    end

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

function no_brancher!(walkers, rng::AbstractRNG)
end