include("wave_function_types.jl")
include("model.jl")

include("accept_reject.jl")

include("vmc.jl")

include("util.jl")

using LinearAlgebra

hamiltonian_sho(ψ, ψstatus, x) = -0.5*ψstatus.laplacian + 0.5*norm(x)^2*ψstatus.value

ψgauss(x) = exp(-norm(x/√2.0)^2)

ψtrial = WaveFunction(
    ψgauss,
    x -> gradient_fd(ψgauss, x),
    x -> laplacian_fd(ψgauss, x)
)

model = Model(
    hamiltonian_sho,
    ψtrial,
)

nconf = 1000
neq = 100

xstart = [1.0]

ψstatus = init_status(ψtrial, xstart)

run_vmc!(model, xstart, 1e-2, accept_reject_diffuse!, ψstatus, nconf, neq)