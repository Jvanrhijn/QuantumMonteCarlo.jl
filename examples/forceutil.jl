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

    # exact jacobian using automatic differentiation
    j = ForwardDiff.jacobian(warp, x)

    return x̅, det(j)

end


function local_energy_gradient(fwalker, model, eref, x′, ψt′, τ; warp=false)
    walker = fwalker.walker
    x = walker.configuration
    ψ = model.wave_function
    ∇ψ = model.wave_function.gradient
    ψ′ = ψt′
    ∇ψ′ = ψ′.gradient

    local_e(x) = model.hamiltonian_recompute(ψ, x) / ψ.value(x)


    if warp
        x̅, _ = node_warp(x, ψ.value(x), ∇ψ(x), ψ′.value(x), ∇ψ′(x), τ)
    else
        x̅ = x
    end

    el = model.hamiltonian_recompute(ψ, x) / ψ.value(x)
    el′ = hamiltonian_recompute′(ψt′, x̅) / ψt′.value(x̅)
    
    ∇ₐel = (el′ - el) / da

    return ∇ₐel
end

function branching_factor_gradient(fwalker, model, eref, x′, ψt′, τ; warp=false)

    x = fwalker.walker.configuration_old
    x′ = fwalker.walker.configuration

    ψ = model.wave_function

    el(r) = model.hamiltonian_recompute(ψ, r) / ψ.value(r)
    els(r) = hamiltonian_recompute′(ψt′, r) / ψt′.value(r)

    # branching factors, for primary and secondary geometry
    s(r) = eref - el(r)
    ss(r) = eref - els(r)

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

function greens_function_gradient(fwalker, model, eref, x′, ψt′, τ; usepq=false, warp=false)
    walker = fwalker.walker
    x = walker.configuration_old
    ψ = model.wave_function

    accepted = x′ == fwalker.walker.configuration

    ψold = ψ.value(x)
    ψproposed = ψ.value(x′)

    node_reject = sign(ψold) != sign(ψproposed)

    if !usepq
        x′ = walker.configuration
    end

    ψnew = ψ.value(x′)
    ψnew′ = ψt′.value(x′)

    el(r) = model.hamiltonian_recompute(ψ, r) / ψ.value(r)
    els(r) = hamiltonian_recompute′(ψt′, r) / ψt′.value(r)

    # drift velocity
    #v(r) = ψ.gradient(r) / ψ.value(r)
    #vs(r) = ψt′.gradient(r) / ψt′.value(r)
    v(r) = QuantumMonteCarlo.cutoff_velocity(ψ.gradient(r) / ψ.value(r), τ)
    vs(r) = QuantumMonteCarlo.cutoff_velocity(ψt′.gradient(r) / ψt′.value(r), τ)

    # drift-diffusion greens function
    t(r′, r) = exp(-norm(r′ - r - v(r)*τ)^2 / 2τ)
    ts(r′, r) = exp(-norm(r′ - r - vs(r)*τ)^2 / 2τ)

    # branching factors
    s(r′, r) = eref - 0.5(el(r) + el(r′))
    ss(r′, r) = eref - 0.5(els(r) + els(r′))

    # full greens function
    g(r′, r) = t(r′, r) * exp(τ * s(r′, r))
    gs(r′, r) = ts(r′, r) * exp(τ * ss(r′, r))
    gb(r′, r) = exp(τ * s(r′, r))
    gbs(r′, r) = exp(τ * ss(r′, r))

    # acceptance and rejection probabilities.
    # The factor at the end ensures that p = 0 when the move x -> x′
    # crosses a node.
    p(r′, r) = min(1, (ψ.value(r′) / ψ.value(r))^2 * t(r, r′) / t(r′, r)) #* Float64(sign(ψ.value(r′)) == sign(ψ.value(r)))
    q(r′, r) = 1 - p(r′, r)
    ps(r′, r) = min(1, (ψt′.value(r′) / ψt′.value(r))^2 * ts(r, r′) / ts(r′, r)) # * Float64(sign(ψt′.value(r)) == sign(ψt′.value(r′)))
    qs(r′, r) = 1 - ps(r′, r)

    # perform warp
    if warp
        x̅, _ = node_warp(x, ψ.value(x), ψ.gradient(x), ψt′.value(x), ψt′.gradient(x), τ)
        x̅′, _ = node_warp(x′, ψ.value(x′), ψ.gradient(x′), ψt′.value(x′), ψt′.gradient(x′), τ)
    else
        x̅ = x
        x̅′ = x′
    end

    if usepq
        if accepted
            deriv = (log(ps(x̅′, x̅) * gs(x̅′, x̅)) - log(p(x′, x) * g(x′, x))) / da 
        elseif node_reject
             deriv = (log(gs(x̅′, x̅)) - log(g(x′, x))) / da
        else
             deriv = (log(qs(x̅′, x̅) * gs(x̅′, x̅)) - log(q(x′, x) * g(x′, x))) / da
        end
    else
        deriv = accepted ? (log(gs(x̅′, x̅)) - log(g(x′, x))) / da : 0.0
    end

    return deriv

end

function jacobian_gradient_previous(fwalker, model, eref, x′, ψt′,  τ)
    walker = fwalker.walker

    x = walker.configuration_old

    accepted = x != walker.configuration

    ψ = model.wave_function.value(x)
    ∇ψ = model.wave_function.gradient(x)

    ψ′ = ψt′.value(x)
    ∇ψ′ = ψt′.gradient(x)

    #x̅, jac = node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′, τ)
    x̅, jac = node_warp_exact_jacobian(x, model.wave_function, ψt′, τ)

    return log(abs(jac)) / da
end

function jacobian_gradient_current(fwalker, model, eref, x′, ψt′,  τ)
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

function log_psi_gradient(fwalker, model, eref, x′, ψt′, τ; warp=false)
    walker = fwalker.walker

    x = walker.configuration

    ψ = model.wave_function.value(x)
    ψ′ = ψt′.value(x)

    ∇ψ = model.wave_function.gradient(x)
    ∇ψ′ = ψt′.gradient(x)

    if warp
        x̅, _ = node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′, τ)
    else
        x̅ = x
    end

    deriv = (log(abs(ψt′.value(x̅))) - log(abs(ψ))) / da

    return deriv 
end