using Random
using StatsBase
using LinearAlgebra
using ProgressMeter
using HDF5
using Formatting
using Dates


function run_dmc!(model, fat_walkers, τ, num_blocks, steps_per_block, eref; rng=MersenneTwister(0), neq=0, outfile=Nothing, verbosity=:silent, brancher=stochastic_reconfiguration!, branchtime=10, dmc=true, accept_reject=DiffuseAcceptReject)
    nwalkers = length(fat_walkers)
    trial_energy = eref

    accumulator = Accumulator(fat_walkers)

    energy_estimate = zeros(num_blocks + 1)
    vmc_estimate = zeros(num_blocks + 1)
    error_estimate = zeros(num_blocks + 1)
    variance_estimate = zeros(num_blocks + 1)
    energy_estimate[1] = eref
    vmc_estimate[1] = eref

    total_weight = 1

    # open output file
    if outfile != Nothing
        if isfile(outfile)
            rm(outfile)
        end
        file = h5open(outfile, "w")
        close(file)
    end

    if verbosity == :progressbar
        progress = Progress(num_blocks + neq)
    end

    start_time = now()

    for j = 1:(num_blocks + neq)

        block_energy = zeros(steps_per_block)
        block_weight = zeros(steps_per_block)
        block_vmc_energy = zeros(steps_per_block)

        for b = 1:steps_per_block
            local_energy_ensemble = zeros(length(fat_walkers))
            weight_ensemble = zeros(length(fat_walkers))
            
            # TODO thread-safe parallelism
            for (i, fwalker) in collect(enumerate(fat_walkers))
                walker = fwalker.walker
                el = model.hamiltonian(walker.ψstatus, walker.configuration) / walker.ψstatus.value

                # perform drift-diffuse step
                accept_reject.move!(walker, τ, model.wave_function, rng)

                x′ = deepcopy(walker.configuration)

                p = accept_reject.compute_acceptance!(walker, τ)
                q = 1 - p

                # accept or reject move
                accept_reject.accept_move!(walker, p, rng)

                el′ = model.hamiltonian(walker.ψstatus, walker.configuration) / walker.ψstatus.value

                v = walker.ψstatus_old.gradient / walker.ψstatus_old.value
                v′ = walker.ψstatus.gradient / walker.ψstatus.value

                ebest = j > neq ? energy_estimate[j-neq] : energy_estimate[1]

                s = (eref - ebest) + (ebest - el) * norm(cutoff_velocity(v, τ)) / norm(v)
                s′ = (eref - ebest) + (ebest- el′) * norm(cutoff_velocity(v′, τ)) / norm(v′)
                #s = (ebest - el) * norm(cutoff_velocity(v, τ)) / norm(v)
                #s′ = (ebest- el′) * norm(cutoff_velocity(v′, τ)) / norm(v′)
                exponent = 0.5 * τ * (s + s′)

                if dmc
                    walker.weight *= exp(exponent)
                else
                    walker.weight = 1.0
                end

                local_energy_ensemble[i] = el′
                weight_ensemble[i] = walker.weight

                # update FatWalker with observables computed at latest
                # configuration
                # pass in also x′ (i.e. the proposed move), which may or
                # may not be equal to x_new
                if j > neq
                    #accumulate_observables!(fwalker, model, eref, x′)
                    accumulate_observables!(fwalker, model, ebest, x′)
                end

            end

            if j > neq
                average_ensemble!(fat_walkers, accumulator)
            end

            ensemble_energy = mean(local_energy_ensemble, Weights(weight_ensemble))

            block_energy[b] = ensemble_energy
            block_weight[b] = mean(weight_ensemble)
            block_vmc_energy[b] = mean(local_energy_ensemble)

            if b % branchtime == 0
                # perform branching
                brancher(fat_walkers, rng)
            end

        end

        if j > neq
            average_block!(accumulator)

            if outfile != Nothing
                file = h5open(outfile, "r+")
                write_to_file!(accumulator, file)
                close(file)
            end
        end

        block_energy = mean(block_energy, Weights(block_weight))
        block_weight = mean(block_weight)
        block_vmc_energy = mean(block_vmc_energy)

        if j <= neq
            eref = energy_estimate[1] - log(block_weight)
        end


        # only update energy esimate after block has run
        if j > neq
            n = j - neq

            s = variance_estimate[n] * total_weight
            total_weight += block_weight
            energy_estimate[n+1] = energy_estimate[n] + block_weight / total_weight * (block_energy - energy_estimate[n])
            vmc_estimate[n+1] = vmc_estimate[n] + 1 / n * (block_vmc_energy - vmc_estimate[n])
            variance_estimate[n+1] = (s + block_weight * (block_energy - energy_estimate[n])*(block_energy - energy_estimate[n+1])) / total_weight
       
            error_estimate[n+1] = sqrt(variance_estimate[n+1] / n)                

            eref = energy_estimate[n+1] - log(block_weight)

        end

        if verbosity == :progressbar
            ProgressMeter.next!(progress)
        elseif verbosity == :loud
            if j > neq
                energy = energy_estimate[j-neq+1]
                vmc_energy = vmc_estimate[j-neq+1]
                err = error_estimate[j-neq+1]
            else
                energy = first(energy_estimate)
                vmc_energy = first(vmc_estimate)
                err = 0.0
            end
            time_elapsed = now() - start_time
            printfmt("Time elapsed: {} | Block: {}/{} | Energy estimate: {:.5f} +- {:.5f} | VMC estimate: {:.5f} | Block energy: {:.5f} | Reference energy: {:.5f} | Trial energy: {:.5f} | Total weight {:.5f} | Block weight {:.5f} \n",
                format_duration(time_elapsed, "HH:MM:SS"),
                lpad(string(j), num_digits(num_blocks + neq), '0'),
                num_blocks + neq,
                energy,
                err,
                vmc_energy,
                block_energy,
                eref,
                first(energy_estimate),
                total_weight,
                block_weight
            )
            flush(stdout)
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
