function run_vmc!(model, fat_walkers, τ, num_blocks, steps_per_block; rng=MersenneTwister(0), neq=0, outfile=Nothing, verbosity=:silent, accept_reject=DiffuseAcceptReject)
    eref = 0.0
    run_dmc!(model, fat_walkers, τ, num_blocks, steps_per_block, eref; rng=rng, neq=neq, outfile=outfile, verbosity=verbosity, brancher=no_brancher!, dmc=false, accept_reject=accept_reject)
end