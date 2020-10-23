function run_vmc!(initial_conf, model, τ, num_blocks, steps_per_block; rng=MersenneTwister(0), neq=0, outfile=Nothing, verbosity=:silent, observables=Nothing)
    init_ψ = WaveFunctionStatus(
        model.wave_function.value(initial_conf),
        model.wave_function.gradient(initial_conf),
        model.wave_function.laplacian(initial_conf)
    )
    walker = Walker(initial_conf, init_ψ)

    if observables == Nothing
        fat_walkers = [FatWalker(walker)]
    else
        fat_walkers = [FatWalker(walker, observables)]
    end
 
    run_dmc!(model, fat_walkers, τ, num_blocks, steps_per_block, eref; rng=rng, neq=neq, outfile=outfile, verbosity=verbosity, brancher=no_brancher!)

end