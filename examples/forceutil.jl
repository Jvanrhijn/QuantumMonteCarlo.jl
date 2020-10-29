using ForwardDiff


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
    x̅ = x .+ (d - d′) * u * sign(ψ′) * n′
   
    # approximate jacobian
    jac = 1 - u + sign(ψ′*ψ) * dot(n, n′) * (u + (d - d′)*uderiv)

    x̅, jac
end

function node_warp_exact_jacobian(x, ψ, ψ′, τ)
    d(y) = abs(ψ.value(y)) / norm(ψ.gradient(y))
    d′(y) = abs(ψ′.value(y)) / norm(ψ′.gradient(y))
    n′(y) = ψ′.gradient(y) / norm(ψ′.gradient(y))

    warp(y::AbstractVector) = y + (d(y) - d′(y)) * n′(y) * sign(ψ′.value(y)) * cutoff_tanh(d(y), a=√τ)[1]

    x̅ = warp(x)

    j = ForwardDiff.jacobian(warp, x)

    return x̅, det(j)

end


function gradel(fwalker, model, eref, x′, ψt′, τ; warp=false)
    walker = fwalker.walker
    x = walker.configuration
    ψ = model.wave_function
    ∇ψ = model.wave_function.gradient
    ψ′ = ψt′
    ∇ψ′ = ψ′.gradient

    local_e(x) = model.hamiltonian_recompute(ψ, x) / ψ.value(x)

    el = model.hamiltonian_recompute(ψ, x) / ψ.value(x)
    el′ = hamiltonian_recompute′(ψt′, x) / ψt′.value(x)

    dx = 1e-5
    ∂ₓel = (local_e(x .+ dx) - local_e(x .- dx)) / 2dx

    if warp
        x̅, _ = node_warp(x, ψ.value(x), ∇ψ(x), ψ′.value(x), ∇ψ′(x), τ)
        ∂ₐx̅ = (x̅ - x)[1] / da
    else
        ∂ₐx̅ = 0
    end
    
    ∇ₐel = (el′ - el) / da
    #println("$deriv")

    return ∇ₐel + ∂ₐx̅ * ∂ₓel
end

function grads(fwalker, model, eref, x′, ψt′, τ; warp=false)

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

    # perform warp
    if warp
        x̅, _ = node_warp(x, ψ.value(x), ψ.gradient(x), ψt′.value(x), ψt′.gradient(x), τ)
        x̅′, _ = node_warp(x′, ψ.value(x′), ψ.gradient(x′), ψt′.value(x′), ψt′.gradient(x′), τ)
    else
        x̅ = x
        x̅′ = x′
    end

    deriv = (Ss(x̅′, x̅) - S(x′, x)) / da

    return deriv

end

function gradt(fwalker, model, eref, x′, ψt′, τ; usepq=false, warp=false)
    walker = fwalker.walker
    x = walker.configuration_old

    accepted = x′ == fwalker.walker.configuration

    if !usepq
        x′ = walker.configuration
    end

    ψ = model.wave_function
    ψnew = ψ.value(x′)
    ψnew′ = ψt′.value(x′)

    el(r) = model.hamiltonian_recompute(ψ, r) / ψ.value(r)
    els(r) = hamiltonian_recompute′(ψt′, r) / ψt′.value(r)

    v(r) = ψ.gradient(r) / ψ.value(r)
    vs(r) = ψt′.gradient(r) / ψt′.value(r)

    t(r′, r) = exp(-norm(r′ - r - v(r)*τ)^2 / 2τ)
    ts(r′, r) = exp(-norm(r′ - r - vs(r)*τ)^2 / 2τ)

    s(r′, r) = eref - 0.5(el(r) + el(r′))
    ss(r′, r) = eref - 0.5(els(r) + els(r′))

    g(r′, r) = t(r′, r) * exp(τ * s(r′, r))
    gs(r′, r) = ts(r′, r) * exp(τ * ss(r′, r))

    p(r′, r) = min(1, (ψ.value(r′) / ψ.value(r))^2 * t(r, r′) / t(r′, r)) * Float64(sign(ψ.value(r′)) == sign(ψ.value(r)))
    q(r′, r) = 1 - p(r′, r)
    ps(r′, r) = min(1, (ψt′.value(r′) / ψt′.value(r))^2 * ts(r, r′) / ts(r′, r)) * Float64(sign(ψt′.value(r)) == sign(ψt′.value(r′)))
    qs(r′, r) = 1 - ps(r′, r)

    # perform warp
    if warp
        x̅, _ = node_warp(x, ψ.value(x), ψ.gradient(x), ψt′.value(x), ψt′.gradient(x), τ)
        x̅′, _ = node_warp(x′, ψ.value(x′), ψ.gradient(x′), ψt′.value(x′), ψt′.gradient(x′), τ)
    else
        x̅ = x
        x̅′ = x′
    end

    if usepq && !accepted
        #deriv = (log(abs(ps(x̅′, x̅) * ts(x̅′, x̅) + qs(x̅′, x̅))) - log(abs(p(x′, x) * t(x′, x) + q(x′, x)))) / da
        #deriv = (ps(x̅′, x̅)*log(ts(x̅′, x̅)) + qs(x̅′, x̅)*log(ts(x̅, x̅)) - (p(x′, x)*log(t(x′, x)) + q(x′, x)*log(t(x, x)))) / da
        #deriv = (log(ps(x̅′, x̅) * gs(x̅′, x̅) + qs(x̅′, x̅)) - log(p(x′, x) * g(x′, x) + q(x′, x))) / da
        deriv = (ps(x̅′, x̅) * (log(ts(x̅′, x̅)) + τ*ss(x̅′, x̅)) + qs(x̅′, x̅) - (p(x′, x) * (log(t(x′, x)) + s(x′, x) * τ) + q(x′, x))) / da
        #deriv = (log(ps(x̅′, x̅) * gs(x̅′, x̅) + qs(x̅′, x̅)) - log(p(x′, x) * g(x′, x) + q(x′, x))) / da
    elseif usepq && accepted
        deriv = (log(gs(x̅′, x̅)) - log(g(x′, x))) / da
    #    #deriv = ((log(ts(x̅′, x̅)) + τ*ss(x̅′, x̅)) - log(t(x′, x)) - τ*s(x′, x)) / da
   # else
   #     deriv = 0
    elseif !warp
        #deriv = accepted ? (log(gs(x′, x)) - log(g(x′, x))) / da : 0.0
        #deriv = accepted ? (log(gs(x′, x)) - log(g(x′, x))) / da : (ss(x, x) - s(x, x)) / da * τ
        ∇ₐv = (vs(x) - v(x)) / da
        u = x′ - x - v(x)*τ
        #∇t = dot(u, ∇ₐv)
        ∇t = dot(u, ∇ₐv)
        ∇s = (ss(x′, x) - s(x′, x)) / da * τ
        deriv = accepted ? (log(gs(x′, x)) - log(g(x′, x))) / da : ∇s
    elseif warp
        ∂ₐlnG = (log(gs(x′, x)) - log(g(x′, x))) / da
        ∂ₐS = (ss(x′, x) - s(x′, x)) / da * τ
        deriv = accepted ? ∂ₐlnG : ∂ₐS
    end

    return deriv

end

function pulay_force_warp_correction_exact(fwalker, model, eref, xp, ψt′, τ)

    x = fwalker.walker.configuration_old

    accepted = xp == fwalker.walker.configuration

    x′ = fwalker.walker.configuration

    ψ = model.wave_function
    x̅′, jac′ = node_warp(x′, ψ.value(x′), ψ.gradient(x′), ψt′.value(x′), ψt′.gradient(x′), τ)
    x̅, jac = node_warp(x, ψ.value(x), ψ.gradient(x), ψt′.value(x), ψt′.gradient(x), τ)

    el(r) = model.hamiltonian_recompute(ψ, r) / ψ.value(r)
    els(r) = hamiltonian_recompute′(ψt′, r) / ψt′.value(r)

    v(r) = ψ.gradient(r) / ψ.value(r)
    vs(r) = ψt′.gradient(r) / ψt′.value(r)

    t(r′, r) = exp(-norm(r′ - r - v(r)*τ)^2 / 2τ)
    ts(r′, r) = exp(-norm(r′ - r - vs(r)*τ)^2 / 2τ)

    s(r′, r) = eref - 0.5(el(r) + el(r′))
    ss(r′, r) = eref - 0.5(els(r) + els(r′))

    g(r′, r) = t(r′, r) * exp(τ * s(r′, r))
    gs(r′, r) = ts(r′, r) * exp(τ * ss(r′, r))

    dx = 1e-5

    ∂ₓlnG′ = (log(g(x′ .+ dx, x)) - log(g(x′ .- dx, x))) / 2dx
    ∂ₓS′ = (ss(x′ .+ dx, x) - s(x′ .- dx, x)) / 2dx * τ

    ∂ₓlnG = (log(g(x′, x .+ dx)) - log(g(x′, x .- dx))) / 2dx
    ∂ₓS = (ss(x′, x .+ dx) - s(x′, x .- dx)) / 2dx * τ

    ∂ₐx̅′ = (x̅′ - x′) / da
    ∂ₐx̅ = (x̅ - x) / da

    return accepted ? ∂ₐx̅′[1] * ∂ₓlnG′ : (∂ₐx̅′[1] * ∂ₓS′)

end

function gradj(fwalker, model, eref, x′, ψt′,  τ)
    walker = fwalker.walker

    x = walker.configuration_old
    #x = walker.configuration

    ψ = model.wave_function.value(x)
    ∇ψ = model.wave_function.gradient(x)

    ψ′ = ψt′.value(x)
    ∇ψ′ = ψt′.gradient(x)

    #x̅, jac = node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′, τ)
    x̅, jac = node_warp_exact_jacobian(x, model.wave_function, ψt′, τ)

    return log(abs(jac)) / da
end

function gradj_last(fwalker, model, eref, x′, ψt′,  τ)
    walker = fwalker.walker

    #x = walker.configuration_old
    x = walker.configuration

    ψ = model.wave_function.value(x)
    ∇ψ = model.wave_function.gradient(x)

    ψ′ = ψt′.value(x)
    ∇ψ′ = ψt′.gradient(x)

    #x̅, jac = node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′, τ)
    x̅, jac = node_warp_exact_jacobian(x, model.wave_function, ψt′, τ)

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

function grad_logpsi_old(fwalker, model, eref, x′)
    return first(fwalker.data["grad log psi hist"])
end

function grad_logpsi_old_warp(fwalker, model, eref, x′)
    return first(fwalker.data["grad log psi hist (warp)"])
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
