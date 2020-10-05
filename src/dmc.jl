using Random
using StatsBase
using LinearAlgebra
using ProgressMeter
using HDF5


function run_dmc!(model, fat_walkers, τ, num_blocks, steps_per_block, eref; rng=MersenneTwister(0), neq=0, outfile=Nothing)
    nwalkers = length(fat_walkers)

    accumulator = Accumulator(fat_walkers)

    energy_estimate = zeros(num_blocks + 1)
    error_estimate = zeros(num_blocks + 1)
    variance_estimate = zeros(num_blocks + 1)
    energy_estimate[1] = eref

    total_weight = nwalkers * steps_per_block

    # open output file
    if outfile != Nothing
        file = h5open(outfile, "w")
    end

    @showprogress for j = 1:(num_blocks + neq)

        block_energy = zeros(steps_per_block)
        block_weight = zeros(steps_per_block)

        for b = 1:steps_per_block
            local_energy_ensemble = zeros(nwalkers)
            weight_ensemble = zeros(nwalkers)
            
            # TODO thread-safe parallelism
            for (i, fwalker) in collect(enumerate(fat_walkers))
                walker = fwalker.walker
                el = model.hamiltonian(model.wave_function, walker.ψstatus, walker.configuration) / walker.ψstatus.value

                # perform drift-diffuse step
                diffuse_walker!(walker, model.wave_function, rng)

                el′ = model.hamiltonian(model.wave_function, walker.ψstatus, walker.configuration) / walker.ψstatus.value

                # compute branching factor                
                s = (eref - 0.5*(el + el′)) * τ

                # update walker weight
                walker.weight *= exp(s)

                # store local energy
                local_energy_ensemble[i] = el′
                weight_ensemble[i] = walker.weight

                # update FatWalker with observables computed at this
                # configuration
                if j > neq
                    accumulate_observables!(fwalker, model, eref)
                end
            end

            if j > neq
                average_ensemble!(fat_walkers, accumulator)
            end

            ensemble_energy = mean(local_energy_ensemble, Weights(weight_ensemble))

            block_energy[b] = ensemble_energy
            block_weight[b] = sum(weight_ensemble)

            if j <= neq
                eref = 0.5 * (eref + ensemble_energy)
            end

        end

        if j > neq
            average_block!(accumulator)

            if outfile != Nothing
                write_to_file!(accumulator, file)
            end
        end

        block_energy = mean(block_energy, Weights(block_weight))
        block_weight = sum(block_weight)

        # perform branching
        stochastic_reconfiguration!(walkers, rng)

        # only update energy esimate after block has run
        if j > neq
            n = j - neq
            eref = 0.5 * (eref + block_energy)
            
            energy_estimate[n+1] = (total_weight*energy_estimate[n] + block_weight*block_energy) /
                (total_weight + block_weight)

            variance_estimate[n+1] = variance_estimate[n] + 
                (block_weight*(block_energy - energy_estimate[n])*(block_energy - energy_estimate[n+1]) -
                variance_estimate[n]) /
                (total_weight + block_weight)

            error_estimate[n+1] = sqrt(variance_estimate[n+1] / n)                

            total_weight += block_weight

        end

    end

    if outfile != Nothing
        close(file)
    end

    return energy_estimate, error_estimate

end