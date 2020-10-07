# All functions needed for calculating forces
function local_energy(fwalker, model, eref)
    ψstatus = fwalker.walker.ψstatus
    x = fwalker.walker.configuration
    model.hamiltonian(ψstatus, x) / ψstatus.value
end

function local_energy_sec(fwalker, model, eref, ψ′)
    x = fwalker.walker.configuration
    model.hamiltonian_recompute(ψ′, x) / ψ′.value(x)
end

function cutoff_tanh(d; a=0.05)
    b = a/5
    value = 0.5 * (1 + tanh((a - d)/b))
    deriv = -0.5 / b / cosh((a - d)/b)^2
    value, deriv
end

function node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′, τ)
    d = abs(ψ) / norm(∇ψ)
    d′ = abs(ψ′) / norm(∇ψ′)
    n′ = ∇ψ′ / norm(∇ψ′)
    n = ∇ψ / norm(∇ψ)
    u, uderiv = cutoff_tanh(d; a=√τ)
    xwarp = x .+ (d - d′) * u * sign(ψ′) * n′
    jac = 1 - u + sign(ψ′*ψ) * dot(n, n′) * (u + (d - d′)*uderiv)
    xwarp, jac
end

function gradel(fwalker, model, eref, ψ′)
    el = fwalker.data["Local energy"]
    el′ = fwalker.data["Local energy (secondary)"]
    (el′ - el) / da
end

function gradel_warp(fwalker, model, eref, ψt′, τ)
    walker = fwalker.walker
    ψ = walker.ψstatus.value
    ∇ψ = walker.ψstatus.gradient
    ψ′ = last(fwalker.data["ψ′"])
    ∇ψ′ = last(fwalker.data["∇ψ′"])
    x = walker.configuration
    xwarp, jac = node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′, τ)

    #el = model.hamiltonian(walker.ψstatus, x) / ψ
    el = last(fwalker.data["Local energy"])
    el′ = model.hamiltonian_recompute(ψt′, xwarp) / ψt′.value(xwarp)
    (el′ - el) / da
end

function psi_sec(fwalker, model, eref, ψ′)
    ψ′.value(fwalker.walker.configuration)
end

function gradpsi_sec(fwalker, model, eref, ψ′)
    ψ′.gradient(fwalker.walker.configuration)
end

function psi_sec(fwalker, model, eref, ψ′)
    ψ′.value(fwalker.walker.configuration_old)
end

function gradpsi_sec_old(fwalker, model, eref, ψ′)
    ψ′.gradient(fwalker.walker.configuration_old)
end

function psi_sec_old(fwalker, model, eref, ψ′)
    ψ′.value(fwalker.walker.configuration_old)
end

function gradpsi_sec_prev(fwalker, model, eref, ψ′)
    ψ′.gradient(fwalker.walker.configuration_old)
end

function grads(fwalker, model, eref, ψ′, τ)
    walker = fwalker.walker
    ∇ₐel = last(fwalker.data["grad el"])
    xprev = fwalker.walker.configuration_old
    el_prev = model.hamiltonian(walker.ψstatus_old, xprev) / walker.ψstatus_old.value
    el_prev′ = model.hamiltonian_recompute(ψ′, xprev) / ψ′.value(xprev)
    ∇ₐel_prev = (el_prev′ .- el_prev) / da
    return -0.5 * (∇ₐel .+ ∇ₐel_prev) * τ
end

function grads_warp(fwalker, model, eref, ψt′, τ)
    walker = fwalker.walker

    ∇ₐel_warp = last(fwalker.data["grad el (warp)"])

    x = walker.configuration
    xprev = walker.configuration_old

    ψ = walker.ψstatus.value
    ∇ψ = walker.ψstatus.gradient

    ψ′ = last(fwalker.data["ψ′"])
    ∇ψ′ = last(fwalker.data["∇ψ′"])

    ψprev = walker.ψstatus_old.value
    ∇ψprev = walker.ψstatus_old.gradient

    ψ′prev = last(fwalker.data["ψ′_old"])
    ∇ψ′prev = last(fwalker.data["∇ψ′_old"])

    # perform warp
    xwarp , _ = node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′, τ)
    xwarpprev, _ = node_warp(xprev, ψprev, ∇ψprev, ψ′prev, ∇ψ′prev, τ)

    el_prev = model.hamiltonian(walker.ψstatus_old, xprev) / walker.ψstatus_old.value
    el_prev′ = model.hamiltonian_recompute(ψt′, xwarpprev) / ψt′.value(xwarpprev)

    ∇ₐel_prev_warp = (el_prev′ - el_prev) / da

    return -0.5 * (∇ₐel_warp + ∇ₐel_prev_warp) * τ
end

function gradt(fwalker, model, eref, ψ′, τ)
    ∇ψ = fwalker.walker.ψstatus_old.gradient
    ψ = fwalker.walker.ψstatus_old.value
    v = ∇ψ / ψ

    x′ = fwalker.walker.configuration
    x = fwalker.walker.configuration_old

    ∇ψsec = ψ′.gradient(x)
    ψsec = ψ′.value(x)
    vsec = ∇ψsec / ψsec

    u = x′ - x - v*τ

    ∇ₐv = (vsec - v) / da

    return dot(u, ∇ₐv)

    #t = -1/(2.0τ) * norm(x′ - x - v*τ)^2
    #t′ = -1/(2.0τ) * norm(x′ - x - vsec*τ)^2

    #return (t' - t) / da
end

function gradt_warp(fwalker, model, eref, ψt′, τ)
    walker = fwalker.walker

    x = walker.configuration
    xprev = walker.configuration_old

    ψ = walker.ψstatus.value
    ∇ψ = walker.ψstatus.gradient

    ψ′ = last(fwalker.data["ψ′"])
    ∇ψ′ = last(fwalker.data["∇ψ′"])

    ψprev = walker.ψstatus_old.value
    ∇ψprev = walker.ψstatus_old.gradient
    v = ∇ψprev / ψprev

    ψ′prev = last(fwalker.data["ψ′_old"])
    ∇ψ′prev = last(fwalker.data["∇ψ′_old"])

    # perform warp
    xwarp , _ = node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′, τ)
    xwarpprev, _ = node_warp(xprev, ψprev, ∇ψprev, ψ′prev, ∇ψ′prev, τ)

    # compute warped drift
    ∇ψ′_old_warp = ψt′.gradient(xwarpprev)
    ψ′_old_warp = ψt′.value(xwarpprev)
    vsec_warp = ∇ψ′_old_warp / ψ′_old_warp

    u = x - xprev - v*τ

    ∇ₐv = (vsec_warp - v) / da

    return dot(u, ∇ₐv)

    #t = -1/(2.0τ) * norm(x .- xprev .- v*τ)^2
    #t′ = -1/(2.0τ) * norm(xwarp .- xwarpprev .- vsec_warp*τ)^2

    #return (t' - t) / da
end

function gradj(fwalker, model, eref, τ)
    walker = fwalker.walker

    x = walker.configuration

    ψ = walker.ψstatus.value
    ∇ψ = walker.ψstatus.gradient

    ψ′ = last(fwalker.data["ψ′"])
    ∇ψ′ = last(fwalker.data["∇ψ′"])

    _, jac = node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′, τ)

    return log(abs(jac)) / da
end

function grad_logpsi(fwalker, model, eref)
    ψ′ = last(fwalker.data["ψ′"])
    ψ = fwalker.walker.ψstatus.value
    (log(abs(ψ′)) - log(abs(ψ))) / da
end

function grad_logpsi_warp(fwalker, model, eref, ψt′, τ)
    walker = fwalker.walker

    x = walker.configuration

    ψ = walker.ψstatus.value
    ∇ψ = walker.ψstatus.gradient

    ψ′ = last(fwalker.data["ψ′"])
    ∇ψ′ = last(fwalker.data["∇ψ′"])

    xwarp, _ = node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′, τ)

    (log(abs(ψt′.value(xwarp))) - log(abs(ψ))) / da
end


function get_weights(fname)
    h5open(fname, "r") do file
        ws = read(file, "Weight")  
        return ws
    end
end

function hellmann_feynman_force(fname)
    h5open(fname, "r") do file
        ∇ₐel = read(file, "grad el")
        ∇ₐel_warp = read(file, "grad el (warp)")
        return -∇ₐel, -∇ₐel_warp
    end
end

function pulay_force_vd(fname)
    h5open(fname, "r") do file
        els = read(file, "Local energy")
        ws = read(file, "Weight")

        energy = mean(els, Weights(ws))

        ∇ₐlogψ = read(file, "grad log psi")
        el∇ₐlogψ = read(file, "Local energy * grad log psi")

        ∇ₐlogψwarp = read(file, "grad log psi (warp)")
        el∇ₐlogψwarp = read(file, "Local energy * grad log psi (warp)")

        ∇ₐlogj = read(file, "grad log j")
        el∇ₐlogj = read(file, "Local energy * grad log j")

        return (
            -2.0*(el∇ₐlogψ - energy*∇ₐlogψ),
            -2.0*(el∇ₐlogψwarp - energy*∇ₐlogψwarp .+
                  el∇ₐlogj - energy*∇ₐlogj)
            )
    end
end

function pulay_force_exact(fname)
    h5open(fname, "r") do file
        els = read(file, "Local energy")
        ws = read(file, "Weight")

        energy = mean(els, Weights(ws))

        Σ∇ₐs = read(file, "grad s")
        elΣ∇ₐs = read(file, "Local energy * grad s")

        Σ∇ₐt = read(file, "grad t")
        elΣ∇ₐt = read(file, "Local energy * grad t")

        Σ∇ₐswarp = read(file, "grad s (warp)")
        elΣ∇ₐswarp = read(file, "Local energy * grad s (warp)")

        Σ∇ₐtwarp = read(file, "grad t (warp)")
        elΣ∇ₐtwarp = read(file, "Local energy * grad t (warp)")

        Σ∇ₐlogj = read(file, "sum grad log j")
        elΣ∇ₐlogj = read(file, "Local energy * sum grad log j")

        fp = -(
            elΣ∇ₐs - energy*Σ∇ₐs .+ 
            elΣ∇ₐt - energy*Σ∇ₐt
        )

        fpwarp = -(
            elΣ∇ₐswarp - energy*Σ∇ₐswarp .+ 
            elΣ∇ₐtwarp - energy*Σ∇ₐtwarp .+ 
            elΣ∇ₐlogj - energy*Σ∇ₐlogj
        )

        return (fp, fpwarp)

    end
end
