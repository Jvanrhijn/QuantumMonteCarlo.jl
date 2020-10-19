# All functions needed for calculating forces
function local_energy(fwalker, model, eref, x′)
    ψstatus = fwalker.walker.ψstatus
    x = fwalker.walker.configuration
    model.hamiltonian(ψstatus, x) / ψstatus.value
end

function local_energy_sec(fwalker, model, eref, x′, ψ′)
    x = fwalker.walker.configuration
    hamiltonian_recompute′(ψ′, x) / ψ′.value(x)
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

function gradel(fwalker, model, eref, x′, ψ′)
    el = fwalker.data["Local energy"]
    el′ = fwalker.data["Local energy (secondary)"]
    (el′ - el) / da
end

function gradel_warp(fwalker, model, eref, x′, ψt′, τ)
    walker = fwalker.walker
    ψ = walker.ψstatus.value
    ∇ψ = walker.ψstatus.gradient
    ψ′ = last(fwalker.data["ψ′"])
    ∇ψ′ = last(fwalker.data["∇ψ′"])
    x = walker.configuration

    xwarp, jac = node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′, τ)

    el = last(fwalker.data["Local energy"])
    el′ = hamiltonian_recompute′(ψt′, xwarp) / ψt′.value(xwarp)
    
    (el′ - el) / da
end

function psi_sec(fwalker, model, eref, x′, ψ′)
    ψ′.value(fwalker.walker.configuration)
end

function gradpsi_sec(fwalker, model, eref, x′, ψ′)
    ψ′.gradient(fwalker.walker.configuration)
end

function gradpsi_sec_old(fwalker, model, eref, x′, ψ′)
    ψ′.gradient(fwalker.walker.configuration_old)
end

function psi_sec_old(fwalker, model, eref, x′, ψ′)
    ψ′.value(fwalker.walker.configuration_old)
end

function grads(fwalker, model, eref, x′, ψt′, τ)

    x = fwalker.walker.configuration_old

    ψ = model.wave_function

    el(r) = model.hamiltonian_recompute(ψ, r) / ψ.value(r)
    els(r) = hamiltonian_recompute(ψt′, r) / ψt′.value(r)

    v(r) = ψ.gradient(r) / ψ.value(r)
    vs(r) = ψt′.gradient(r) / ψt′.value(r)

    t(r′, r) = exp(-norm(r′ - r - v(r)*τ)^2 / 2τ)
    ts(r′, r) = exp(-norm(r′ - r - vs(r)*τ)^2 / 2τ)

    s(r) = eref - el(r)
    ss(r) = eref - els(r)

    p(r′, r) = min(1, (ψ.value(r′) / ψ.value(r))^2 * t(r, r′) / t(r′, r))
    q(r′, r) = 1 - p(r′, r)
    ps(r′, r) = min(1, (ψt′.value(r′) / ψt′.value(r))^2 * ts(r, r′) / ts(r′, r))
    qs(r′, r) = 1 - ps(r′, r)

    S(r′, r) = (0.5p(r′, r) * (s(r′) + s(r)) + q(r′, r) * s(r)) * τ
    Ss(r′, r) = (0.5ps(r′, r) * (ss(r′) + ss(r)) + qs(r′, r) * ss(r)) * τ

    #deriv = (log(abs(ps(x′, x) * ss(x′, x))) - log(abs(p(x′, x) * s(x′, x)))) / da
    deriv = (Ss(x′, x) - S(x′, x)) / da

    return deriv
  
end

function grads_warp(fwalker, model, eref, x′, ψt′, τ)

    x = fwalker.walker.configuration_old

    ψ = model.wave_function

    el(r) = model.hamiltonian_recompute(ψ, r) / ψ.value(r)
    els(r) = hamiltonian_recompute(ψt′, r) / ψt′.value(r)

    v(r) = ψ.gradient(r) / ψ.value(r)
    vs(r) = ψt′.gradient(r) / ψt′.value(r)

    t(r′, r) = exp(-norm(r′ - r - v(r)*τ)^2 / 2τ)
    ts(r′, r) = exp(-norm(r′ - r - vs(r)*τ)^2 / 2τ)

    s(r) = eref - el(r)
    ss(r) = eref - els(r)

    p(r′, r) = min(1, (ψ.value(r′) / ψ.value(r))^2 * t(r, r′) / t(r′, r))
    q(r′, r) = 1 - p(r′, r)
    ps(r′, r) = min(1, (ψt′.value(r′) / ψt′.value(r))^2 * ts(r, r′) / ts(r′, r))
    qs(r′, r) = 1 - ps(r′, r)

    S(r′, r) = (0.5p(r′, r) * (s(r′) + s(r)) + q(r′, r) * s(r)) * τ
    Ss(r′, r) = (0.5ps(r′, r) * (ss(r′) + ss(r)) + qs(r′, r) * ss(r)) * τ

    # perform warp
    x̅, _ = node_warp(x, ψ.value(x), ψ.gradient(x), ψt′.value(x), ψt′.gradient(x), τ)
    x̅′, _ = node_warp(x′, ψ.value(x′), ψ.gradient(x′), ψt′.value(x′), ψt′.gradient(x′), τ)

    #deriv = (log(abs(ps(x′, x) * ss(x′, x))) - log(abs(p(x′, x) * s(x′, x)))) / da
    deriv = (Ss(x̅′, x̅) - S(x̅′, x̅)) / da

    return deriv

end

function gradt(fwalker, model, eref, x′, ψt′, τ)

    x = fwalker.walker.configuration_old

    ψ = model.wave_function

    el(r) = model.hamiltonian_recompute(ψ, r) / ψ.value(r)
    els(r) = hamiltonian_recompute(ψt′, r) / ψt′.value(r)

    v(r) = ψ.gradient(r) / ψ.value(r)
    vs(r) = ψt′.gradient(r) / ψt′.value(r)

    t(r′, r) = exp(-norm(r′ - r - v(r)*τ)^2 / 2τ)
    ts(r′, r) = exp(-norm(r′ - r - vs(r)*τ)^2 / 2τ)

    p(r′, r) = min(1, (ψ.value(r′) / ψ.value(r))^2 * t(r, r′) / t(r′, r))
    q(r′, r) = 1 - p(r′, r)
    ps(r′, r) = min(1, (ψt′.value(r′) / ψt′.value(r))^2 * ts(r, r′) / ts(r′, r))
    qs(r′, r) = 1 - ps(r′, r)

    #deriv = (log(abs(ps(x′, x) * ts(x′, x))) - log(abs(p(x′, x) * t(x′, x)))) / da
    deriv = (log(abs(ps(x′, x) * ts(x′, x) + 1 - ps(x′, x))) - log(abs(p(x′, x) * t(x′, x) + 1 - p(x′, x)))) / da

    return deriv
  
end

function gradt_warp(fwalker, model, eref, x′, ψt′, τ)
    walker = fwalker.walker
    x = walker.configuration_old

    ψ = model.wave_function

    el(r) = model.hamiltonian_recompute(ψ, r) / ψ.value(r)
    els(r) = hamiltonian_recompute(ψt′, r) / ψt′.value(r)

    v(r) = ψ.gradient(r) / ψ.value(r)
    vs(r) = ψt′.gradient(r) / ψt′.value(r)

    t(r′, r) = exp(-norm(r′ - r - v(r)*τ)^2 / 2τ)
    ts(r′, r) = exp(-norm(r′ - r - vs(r)*τ)^2 / 2τ)

    s(r′, r) = τ * (eref - 0.5 * (el(r) + el(r′))) * 0
    ss(r′, r) = τ * (eref - 0.5 * (els(r) + els(r′))) * 0

    g(r′, r) = t(r′, r) * exp(s(r′, r))
    gs(r′, r) = ts(r′, r) * exp(ss(r′, r))

    p(r′, r) = min(1, (ψ.value(r′) / ψ.value(r)) ^2 * t(r, r′) / t(r′, r))
    q(r′, r) = 1 - p(r′, r)
    ps(r′, r) = min(1, (ψt′.value(r′) / ψt′.value(r)) ^2 * ts(r, r′) / ts(r′, r))
    qs(r′, r) = 1 - ps(r′, r)

    # perform warp
    x̅, _ = node_warp(x, ψ.value(x), ψ.gradient(x), ψt′.value(x), ψt′.gradient(x), τ)
    x̅′, _ = node_warp(x′, ψ.value(x′), ψ.gradient(x′), ψt′.value(x′), ψt′.gradient(x′), τ)

    deriv = (log(abs(ps(x̅′, x̅) * ts(x̅′, x̅))) - log(abs(p(x̅′, x̅) * t(x̅′, x̅)))) / da

end

function gradj(fwalker, model, eref, x′, τ)
    walker = fwalker.walker

    x = walker.configuration

    ψ = walker.ψstatus.value
    ∇ψ = walker.ψstatus.gradient

    ψ′ = last(fwalker.data["ψ′"])
    ∇ψ′ = last(fwalker.data["∇ψ′"])

    _, jac = node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′, τ)

    return log(abs(jac)) / da
end

function grad_logpsi(fwalker, model, eref, x′, ψt′)
    ψ′ = last(fwalker.data["ψ′"])
    ψ = fwalker.walker.ψstatus.value
    (log(abs(ψ′)) - log(abs(ψ))) / da
end

function grad_logpsi_warp(fwalker, model, eref, x′, ψt′, τ)
    walker = fwalker.walker

    x = walker.configuration

    ψ = walker.ψstatus.value
    ∇ψ = walker.ψstatus.gradient

    ψ′ = last(fwalker.data["ψ′"])
    ∇ψ′ = last(fwalker.data["∇ψ′"])

    xwarp, _ = node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′, τ)
    deriv = (log(abs(ψt′.value(xwarp))) - log(abs(ψ))) / da
    return deriv 
end

function psi_history(fwalker, model, eref, x′)
    return fwalker.walker.ψstatus_old.value
end

function psi_history′(fwalker, model, eref, x′, ψt′)
    return ψt′.value(fwalker.walker.configuration_old)
end

function grad_logpsisquared_old(fwalker, model, eref, x′)
    ψold = first(fwalker.data["psi history"])
    ψold′ = first(fwalker.data["psi history (secondary)"])
    return (log(abs(ψold′^2)) - log(abs(ψold^2))) / da
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

        return -(
            2.0*(el∇ₐlogψ - energy*∇ₐlogψ),
            2.0*(el∇ₐlogψwarp - energy*∇ₐlogψwarp .+
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
