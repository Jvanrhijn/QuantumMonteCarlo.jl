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

function grads(fwalker, model, eref, x′, ψ′, τ)
    walker = fwalker.walker
    ∇ₐel = last(fwalker.data["grad el"])
    xprev = fwalker.walker.configuration_old
    x = fwalker.walker.configuration


    el = last(fwalker.data["Local energy"])
    el′ = last(fwalker.data["Local energy (secondary)"])

    ψ = walker.ψstatus.value
    ∇ψ = walker.ψstatus.gradient
    v = ∇ψ / ψ

    ψold = walker.ψstatus_old.value
    ∇ψold = walker.ψstatus_old.gradient
    vold = ∇ψold / ψold

    ψsec = last(fwalker.data["ψ′"])
    ∇ψsec = last(fwalker.data["∇ψ′"])
    vsec = ∇ψsec / ψsec

    ψsec_old = last(fwalker.data["ψ′_old"])
    ∇ψsec_old = last(fwalker.data["∇ψ′_old"])
    vsec_old = ∇ψsec_old / ψsec_old

    el_prev = model.hamiltonian_recompute(model.wave_function, xprev) / model.wave_function.value(xprev)
    el_prev′ = hamiltonian_recompute′(ψ′, xprev) / ψ′.value(xprev)

    s = (eref - el)# * ratio
    sprev = (eref - el_prev)# * ratio_old
    S = exp(0.5 * (s + sprev) * τ)

    s′ = (eref - el′)# * ratio_sec
    sprev′ = (eref - el_prev′)# * ratio_sec_old
    S′ = exp(0.5 * (s′ + sprev′) * τ)

    num = exp.(-norm(xprev .- x .- v*τ)^2 / 2τ)
    denom = exp.(-norm(x .- xprev .- vold*τ)^2 / 2τ)

    numsec = exp.(-norm(xprev .- x .- vsec*τ)^2 / 2τ)
    denomsec = exp.(-norm(x .- xprev .- vsec_old*τ)^2 / 2τ)

    p = min(1, ψ^2 / ψold^2 * num / denom)
    psec = min(1, ψsec^2 / ψsec_old^2 * numsec / denomsec)

    #arg = p*S + 1 - p
    #argsec = psec*S′ + 1 - psec
    arg = S
    argsec = S′
    
    (log(abs(argsec)) - log(abs(arg))) / da

end

function grads_warp(fwalker, model, eref, x′, ψt′, τ)
    walker = fwalker.walker

    x = walker.configuration
    xprev = walker.configuration_old

    ψ = walker.ψstatus.value
    ∇ψ = walker.ψstatus.gradient
    v = ∇ψ / ψ

    ψ′ = last(fwalker.data["ψ′"])
    ∇ψ′ = last(fwalker.data["∇ψ′"])
    v′ = ∇ψ′ / ψ′

    ψprev = walker.ψstatus_old.value
    ∇ψprev = walker.ψstatus_old.gradient
    vprev = ∇ψprev / ψprev

    ψ′prev = last(fwalker.data["ψ′_old"])
    ∇ψ′prev = last(fwalker.data["∇ψ′_old"])
    v′prev = ∇ψ′prev / ψ′prev

    # perform warp
    xwarp , _ = node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′, τ)
    xwarpprev, _ = node_warp(xprev, ψprev, ∇ψprev, ψ′prev, ∇ψ′prev, τ)

    el = last(fwalker.data["Local energy"])
    el′ = hamiltonian_recompute′(ψt′, xwarp) / ψt′.value(xwarp)

    el_prev = model.hamiltonian_recompute(model.wave_function, xprev) / model.wave_function.value(xprev)
    el_prev′ = hamiltonian_recompute′(ψt′, xwarpprev) / ψt′.value(xwarpprev)

    sprev = (eref - el_prev)# * ratio_prev
    s = (eref - el)# * ratio
    S = exp(0.5τ * (s + sprev))
    
    sprev′ = (eref - el_prev′)
    s′ = (eref - el′)
    S′ = exp(0.5τ * (sprev′ + s′))

    num = exp.(-norm(xprev .- x .- v*τ)^2 / 2τ)
    denom = exp.(-norm(x .- xprev .- vprev*τ)^2 / 2τ)

    numsec = exp.(-norm(xprev .- x .- v′*τ)^2 / 2τ)
    denomsec = exp.(-norm(x .- xprev .- v′prev*τ)^2 / 2τ)

    p = min(1, ψ^2 / ψprev^2 * num / denom)
    psec = min(1, ψ′^2 / ψ′prev^2 * numsec / denomsec)

#    arg = p*S + 1 - p
#    argsec = psec*S′ + 1 - psec
    arg = S
    argsec = S′

    (log(abs(argsec)) - log(abs(arg))) / da

end

function gradt(fwalker, model, eref, x′, ψ′, τ)

    x = fwalker.walker.configuration_old
    x′ = fwalker.walker.configuration

    ψ = model.wave_function.value(x)
    ∇ψ = model.wave_function.gradient(x)
    v = ∇ψ / ψ

    ∇ψsec = ψ′.gradient(x)
    ψsec = ψ′.value(x)
    vsec = ∇ψsec / ψsec

    ψnew = model.wave_function.value(x′)
    ∇ψnew = model.wave_function.gradient(x′)
    vnew = ∇ψnew / ψnew

    ψnew_sec = ψ′.value(x′)
    ∇ψnew_sec = ψ′.gradient(x′)
    vnew_sec = ∇ψnew_sec / ψnew_sec

    num = exp.(-norm(x .- x′ .- vnew*τ)^2 / 2τ)
    denom = exp.(-norm(x′ .- x .- v*τ)^2 / 2τ)

    numsec = exp.(-norm(x .- x′ .- vnew_sec*τ)^2 / 2τ)
    denomsec = exp.(-norm(x′ .- x .- vsec*τ)^2 / 2τ)

    p = min(1, ψnew^2 / ψ^2 * num / denom)
    psec = min(1, ψnew_sec^2 / ψsec^2 * numsec / denomsec)

    u = x′ - x - v*τ
    usec = x′ - x - vsec*τ

    tsec = exp(-norm(usec)^2 / 2τ)
    tsec_rr = exp(-norm(vsec*τ)^2 / 2τ)
    t = exp(-norm(u)^2 / 2τ)
    t_rr = exp(-norm(v*τ)^2 / 2τ)

    argsec = psec*tsec + (1 - psec)*tsec_rr
    arg = p*t + (1 - p)*t_rr

    (log(abs(argsec)) - log(abs(arg))) / da
  
end

function gradt_warp(fwalker, model, eref, x, ψt′, τ)
    walker = fwalker.walker

    xprev = walker.configuration_old

    ψ = model.wave_function.value(x)
    ∇ψ = model.wave_function.gradient(x)
    v = ∇ψ / ψ

    ψ′ = model.wave_function.value(x)
    ∇ψ′ = model.wave_function.gradient(x)

    ψprev = model.wave_function.value(xprev)
    ∇ψprev = model.wave_function.gradient(xprev)
    vprev = ∇ψprev / ψprev

    ψ′prev = ψtrial′.value(xprev)
    ∇ψ′prev = ψtrial′.gradient(xprev)

    # perform warp
    xwarp , _ = node_warp(x, ψ, ∇ψ, ψ′, ∇ψ′, τ)
    xwarpprev, _ = node_warp(xprev, ψprev, ∇ψprev, ψ′prev, ∇ψ′prev, τ)

    # compute warped drift
    ∇ψ′_old_warp = ψt′.gradient(xwarpprev)
    ψ′_old_warp = ψt′.value(xwarpprev)
    vsec_warp_prev = ∇ψ′_old_warp / ψ′_old_warp
    vsec_warp = ψt′.gradient(xwarp) / ψt′.value(xwarp)

    num = exp.(-norm(xprev .- x .- v*τ)^2 / 2τ)
    denom = exp.(-norm(x .- xprev .- vprev*τ)^2 / 2τ)

    numsec = exp.(-norm(xwarpprev .- xwarp .- vsec_warp*τ)^2 / 2τ)
    denomsec = exp.(-norm(xwarp .- xwarpprev .- vsec_warp_prev*τ)^2 / 2τ)

    p = min(1, ψ^2 / ψprev^2 * num / denom)
    psec = min(1, ψ′^2 / ψ′prev^2 * numsec / denomsec)

    u = x - xprev - vprev*τ
    u′ = xwarp - xwarpprev - vsec_warp_prev*τ

    t′ = exp(-norm(u′)^2 / 2τ)
    t′_rr = exp(-norm(vsec_warp_prev*τ)^2 / 2τ)

    t = exp(-norm(u)^2 / 2τ)
    t_rr = exp(-norm(vprev*τ)^2 / 2τ)

    argsec = psec*t′ + (1 - psec) * t′_rr
    arg = p*t + (1 - p) * t_rr

    (log(abs(argsec)) - log(abs(arg))) / da

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
