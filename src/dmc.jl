using Random
using StatsBase
using LinearAlgebra
using ProgressMeter
using HDF5
using Formatting
using Dates


function run_dmc!(model, fat_walkers, τ, num_blocks, steps_per_block, eref; rng=MersenneTwister(0), neq=0, outfile=Nothing, verbosity=:silent)
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

    if verbosity == :progressbar
        p = Progress(num_blocks + neq)
    end

    start_time = now()

    for j = 1:(num_blocks + neq)

        block_energy = zeros(steps_per_block)
        block_weight = zeros(steps_per_block)

        for b = 1:steps_per_block
            local_energy_ensemble = zeros(nwalkers)
            weight_ensemble = zeros(nwalkers)
            
            # TODO thread-safe parallelism
            for (i, fwalker) in collect(enumerate(fat_walkers))
                walker = fwalker.walker
                el = model.hamiltonian(walker.ψstatus, walker.configuration) / walker.ψstatus.value

                # perform drift-diffuse step
                diffuse_walker!(walker, model.wave_function, τ, rng)

                el′ = model.hamiltonian(walker.ψstatus, walker.configuration) / walker.ψstatus.value

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
        stochastic_reconfiguration!(fat_walkers, rng)

        # only update energy esimate after block has run
        if j > neq
            n = j - neq
            
            energy_estimate[n+1] = (total_weight*energy_estimate[n] + block_weight*block_energy) /
                (total_weight + block_weight)

            variance_estimate[n+1] = variance_estimate[n] + 
                (block_weight*(block_energy - energy_estimate[n])*(block_energy - energy_estimate[n+1]) -
                variance_estimate[n]) /
                (total_weight + block_weight)

            error_estimate[n+1] = sqrt(variance_estimate[n+1] / n)                

            total_weight += block_weight

            eref = 0.5 * (eref + energy_estimate[n+1])

        end

        if verbosity == :progressbar
            ProgressMeter.next!(p)
        elseif verbosity == :loud
            if j > neq
                energy = energy_estimate[j-neq+1]
                err = error_estimate[j-neq+1]
            else
                energy = first(energy_estimate)
                err = 0.0
            end
            time_elapsed = now() - start_time
            printfmt("Time elapsed: {} | Block: {}/{} | Energy estimate: {:.5f} +- {:.5f} | Block energy: {:.5f} | Trial energy: {:.5f}\n",
                format_duration(time_elapsed, "HH:MM:SS"),
                lpad(string(j), num_digits(num_blocks + neq), '0'),
                num_blocks + neq,
                energy,
                err,
                block_energy,
                first(energy_estimate)
            )
        end

    end

    if outfile != Nothing
        close(file)
    end

    return energy_estimate, error_estimate

end

function format_duration(duration, format)
    periods = Dates.canonicalize(Dates.CompoundPeriod(duration))
    date = Dates.DateTime(periods.periods...)
    Dates.format(date, format)
end