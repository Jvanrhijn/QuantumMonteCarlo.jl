# All functions needed for calculating forces
function local_energy(fwalker, model, eref, x′)
    ψ = model.wave_function
    x = fwalker.walker.configuration
    model.hamiltonian_recompute(ψ, x) / ψ.value(x)
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

function gradel(fwalker, model, eref, x′, ψt′)
    x = fwalker.walker.configuration
    ψ = model.wave_function

    el = model.hamiltonian_recompute(ψ, x) / ψ.value(x)
    el′ = hamiltonian_recompute′(ψt′, x) / ψt′.value(x)

    deriv = (el′ - el) / da
    #println("$deriv")
    return deriv
end

function gradel_warp(fwalker, model, eref, x′, ψt′, τ)
    walker = fwalker.walker
    ψ = model.wave_function
    ∇ψ = ψ.gradient
    ψ′ = ψt′
    ∇ψ′ = ψt′.gradient
    x = walker.configuration

    x̅, jac = node_warp(x, ψ.value(x), ∇ψ(x), ψ′.value(x), ∇ψ′(x), τ)

    el = model.hamiltonian_recompute(ψ, x) / ψ.value(x)
    el′ = hamiltonian_recompute′(ψt′, x̅) / ψt′.value(x̅)
    
    deriv = (el′ - el) / da
    #println("$deriv")

    return deriv
end

function grads(fwalker, model, eref, x′, ψt′, τ)

    x = fwalker.walker.configuration_old
    x′ = fwalker.walker.configuration

    ψ = model.wave_function

    el(r) = model.hamiltonian_recompute(ψ, r) / ψ.value(r)
    els(r) = hamiltonian_recompute′(ψt′, r) / ψt′.value(r)

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

    S(r′, r) = 0.5τ * (s(r) + s(r′))
    Ss(r′, r) = 0.5τ * (ss(r) + ss(r′))
    #S(r′, r) = (0.5p(r′, r) * (s(r′) + s(r)) + q(r′, r) * s(r)) * τ
    #Ss(r′, r) = (0.5ps(r′, r) * (ss(r′) + ss(r)) + qs(r′, r) * ss(r)) * τ

    deriv = (Ss(x′, x) - S(x′, x)) / da

    return deriv
  
end

function grads_warp(fwalker, model, eref, x′, ψt′, τ)

    x = fwalker.walker.configuration_old
    x′ = fwalker.walker.configuration

    ψ = model.wave_function

    el(r) = model.hamiltonian_recompute(ψ, r) / ψ.value(r)
    els(r) = hamiltonian_recompute′(ψt′, r) / ψt′.value(r)

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

    S(r′, r) = 0.5 * τ * (s(r) + s(r′))
    Ss(r′, r) = 0.5 * τ * (ss(r) + ss(r′))
    #S(r′, r) = (0.5p(r′, r) * (s(r′) + s(r)) + q(r′, r) * s(r)) * τ
    #Ss(r′, r) = (0.5ps(r′, r) * (ss(r′) + ss(r)) + qs(r′, r) * ss(r)) * τ

    # perform warp
    x̅, _ = node_warp(x, ψ.value(x), ψ.gradient(x), ψt′.value(x), ψt′.gradient(x), τ)
    x̅′, _ = node_warp(x′, ψ.value(x′), ψ.gradient(x′), ψt′.value(x′), ψt′.gradient(x′), τ)

    deriv = (Ss(x̅′, x̅) - S(x′, x)) / da
    #deriv = (Ss(x′, x) - S(x′, x)) / da

    return deriv

end

function gradt(fwalker, model, eref, x′, ψt′, τ, usepq)

    x = fwalker.walker.configuration_old

    ψ = model.wave_function

    ψnew = ψ.value(x′)

    v(r) = ψ.gradient(r) / ψ.value(r)
    vs(r) = ψt′.gradient(r) / ψt′.value(r)

    t(r′, r) = exp(-norm(r′ - r - v(r)*τ)^2 / 2τ)
    ts(r′, r) = exp(-norm(r′ - r - vs(r)*τ)^2 / 2τ)

    p(r′, r) = min(1, (ψ.value(r′) / ψ.value(r))^2 * t(r, r′) / t(r′, r))
    q(r′, r) = 1 - p(r′, r)

    ps(r′, r) = min(1, (ψt′.value(r′) / ψt′.value(r))^2 * ts(r, r′) / ts(r′, r))
    qs(r′, r) = 1 - ps(r′, r)

    if ψnew != 0.0
        if usepq
            deriv = (log(abs(ps(x′, x) * ts(x′, x) + qs(x′, x))) - log(abs(p(x′, x) * t(x′, x) + q(x′, x)))) / da
        else
            x′ = fwalker.walker.configuration
            deriv = (log(abs(ts(x′, x))) - log(abs(t(x′, x)))) / da
        end
        #deriv = (ps(x′, x) * log(abs(ts(x′, x))) - p(x′, x) * log(abs(t(x′, x)))) / da
        #deriv = (ps(x′, x) * log(abs(ts(x′, x))) - p(x′, x) * log(abs(t(x′, x))) + qs(x′, x) * log(abs(ts(x, x))) - q(x′, x) * log(abs(t(x, x)))) / da
    else
        #deriv = 0.0
        deriv = (log(abs(ts(x, x))) - log(abs(t(x, x)))) / da
    end

    return deriv
  
end

function gradt_warp(fwalker, model, eref, x′, ψt′, τ, usepq)
    walker = fwalker.walker
    x = walker.configuration_old

    ψ = model.wave_function
    ψnew = ψ.value(x′)

    el(r) = model.hamiltonian_recompute(ψ, r) / ψ.value(r)
    els(r) = hamiltanian_recompute′(ψt′, r) / ψt′.value(r)

    v(r) = ψ.gradient(r) / ψ.value(r)
    vs(r) = ψt′.gradient(r) / ψt′.value(r)

    t(r′, r) = exp(-norm(r′ - r - v(r)*τ)^2 / 2τ)
    ts(r′, r) = exp(-norm(r′ - r - vs(r)*τ)^2 / 2τ)

    p(r′, r) = min(1, (ψ.value(r′) / ψ.value(r)) ^2 * t(r, r′) / t(r′, r))
    q(r′, r) = 1 - p(r′, r)
    ps(r′, r) = min(1, (ψt′.value(r′) / ψt′.value(r)) ^2 * ts(r, r′) / ts(r′, r))
    qs(r′, r) = 1 - ps(r′, r)

    # perform warp
    x̅, _ = node_warp(x, ψ.value(x), ψ.gradient(x), ψt′.value(x), ψt′.gradient(x), τ)
    x̅′, _ = node_warp(x′, ψ.value(x′), ψ.gradient(x′), ψt′.value(x′), ψt′.gradient(x′), τ)

    if ψnew != 0.0
        if usepq
            deriv = (log(abs(ps(x̅′, x̅) * ts(x̅′, x̅) + qs(x̅′, x̅))) - log(abs(p(x′, x) * t(x′, x) + q(x′, x)))) / da
        else
            x′ = fwalker.walker.configuration
            x̅′, _ = node_warp(x′, ψ.value(x′), ψ.gradient(x′), ψt′.value(x′), ψt′.gradient(x′), τ)
            deriv = (log(abs(ts(x̅′, x̅))) - log(abs(t(x′, x)))) / da
        end
    else
        deriv = (log(abs(ts(x̅, x̅))) - log(abs(t(x, x)))) / da
    end

    return deriv

end

function gradj(fwalker, model, eref, x′, ψt′,  τ)
    walker = fwalker.walker

    x = walker.configuration

    ψ = model.wave_function.value(x)
    ∇ψ = model.wave_function.gradient(x)

    ψ′ = ψt′.value(x)
    ∇ψ′ = ψt′.gradient(x)

    x̅, jac = node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′, τ)

    return log(abs(jac)) / da
end

function grad_logpsi(fwalker, model, eref, x′, ψt′)
    x = fwalker.walker.configuration

    ψ′ = ψt′.value(x)
    ψ = model.wave_function.value(x)

    (log(abs(ψ′)) - log(abs(ψ))) / da
end

function grad_logpsi_warp(fwalker, model, eref, x′, ψt′, τ)
    walker = fwalker.walker

    x = walker.configuration

    ψ = model.wave_function.value(x)
    ψ′ = ψt′.value(x)

    ∇ψ = model.wave_function.gradient(x)
    ∇ψ′ = ψt′.gradient(x)

    x̅, _ = node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′, τ)

    deriv = (log(abs(ψt′.value(x̅))) - log(abs(ψ))) / da

    return deriv 
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
