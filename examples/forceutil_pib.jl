# All functions needed for calculating forces
function local_energy(fwalker, model, eref)
    ψstatus = fwalker.walker.ψstatus
    x = fwalker.walker.configuration
    model.hamiltonian(ψstatus, x) / ψstatus.value
end

function gradel(fwalker, model, eref, ψ′)
    x = fwalker.walker.configuration
    -2a / (a^2 - x[1]^2)^2
end

function gradel_warp(fwalker, model, eref, ψ′)
    x = fwalker.walker.configuration
    -2a / (a^2 - x[1]^2)^2 + x[1] / (a^2 - x[1]^2)^2 * sign(x[1])
end

function grads(fwalker, model, eref, ψt′, τ)
    x = fwalker.walker.configuration_old
    x′ = fwalker.walker.configuration
    ∇ₐel = -2a/(x[1]^2 - a^2)^2
    ∇ₐel′ = -2a/(x′[1]^2 - a^2)^2

    -0.5τ * (∇ₐel + ∇ₐel′)

end

function grads_warp(fwalker, model, eref, ψt′, τ)
    x = fwalker.walker.configuration_old
    x′ = fwalker.walker.configuration
    ∇ₐel = -2a/(x[1]^2 - a^2)^2 + x[1] / (a^2 - x[1]^2)^2 * sign(x[1]) 
    ∇ₐel′ = -2a/(x′[1]^2 - a^2)^2 + x′[1] / (a^2 - x′[1]^2)^2 * sign(x′[1]) 

    -0.5τ * (∇ₐel + ∇ₐel′)
end

function gradt(fwalker, model, eref, ψ′, τ)
    ∇ψ = fwalker.walker.ψstatus_old.gradient
    ψ = fwalker.walker.ψstatus_old.value
    #v = ∇ψ / ψ


    x′ = fwalker.walker.configuration
    x = fwalker.walker.configuration_old

    v = -2x / (a^2 - x[1]^2)


    u = x′ - x - v*τ

    ∇ₐv = 4a*x / (a^2 - x[1]^2)^2

    if x′ == x
        return 0
    end

    return dot(u, ∇ₐv)

end

function gradt_warp(fwalker, model, eref, ψ′, τ)
    ∇ψ = fwalker.walker.ψstatus_old.gradient
    ψ = fwalker.walker.ψstatus_old.value
    #v = ∇ψ / ψ

    x′ = fwalker.walker.configuration
    x = fwalker.walker.configuration_old

    v = -2x / (a^2 - x[1]^2)

    if x′ == x
        return 0
    end

    u = x′ - x - v*τ

    ∇ₐv = 4a*x / (a^2 - x[1]^2)^2
    ∂ₓv = -2/(a^2 - x[1]^2) + 4x[1]^2 / (a^2 - x[1]^2)^2

    ∂ₐT = dot(u, ∇ₐv)
    ∂ₐxw = 0.5*sign(x[1])
    ∂ₐxw′ = 0.5*sign(x′[1])

    ∂ₓ′T = -u[1]/τ
    ∂ₓT = u[1]/τ + u[1] * ∂ₓv

    return ∂ₐT + ∂ₐxw * ∂ₓT + ∂ₐxw′ * ∂ₓ′T

end

function grad_logpsi(fwalker, model, eref, ψt′)
    x = fwalker.walker.configuration
    2a / (a^2 - x[1]^2)
end

function grad_logpsi_warp(fwalker, model, eref, ψt′)
    x = fwalker.walker.configuration
    2a / (a^2 - x[1]^2) - sign(x[1]) * x[1] / (a^2 - x[1]^2)
end

function grad_jac(fwalker, model, eref)
    1 - π^2/8
end