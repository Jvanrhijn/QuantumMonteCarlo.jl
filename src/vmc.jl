using Random
using Statistics


function run_vmc!(model, x, τ, accept_reject!, ψstatus, nconf, neq; seed=0)
    rng = MersenneTwister(seed)

    ψ = model.wave_function

    local_es = zeros(nconf)

    for i = 1:(nconf + neq)
        accept_reject!(ψ, τ, ψstatus, x, rng)
        #accumulator!(ψ, ψ_status)

        #log!(accumulator, ψ, ψ_status)

        if i > neq
            j = i - neq
            local_es[j] = model.hamiltonian(ψ, ψstatus, x) / ψstatus.value
            println("$(local_es[j])     $(mean(local_es[1:j]))")
        end
    end

end