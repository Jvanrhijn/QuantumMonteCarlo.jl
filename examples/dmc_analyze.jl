using QuantumMonteCarlo
using Plots
using HDF5
using StatsBase

include("forceutil.jl")

ws = get_weights(ARGS[1])

flhf, flhf_warp = hellmann_feynman_force("test.hdf5")
flpexact, flpexact_warp = pulay_force_exact("test.hdf5")
flpvd, flpvd_warp = pulay_force_vd("test.hdf5")

fhf = mean(flhf, Weights(ws))
fhf_warp = mean(flhf_warp, Weights(ws))

fpexact = mean(flpexact, Weights(ws))
fpexact_warp = mean(flpexact_warp, Weights(ws))

fpvd = mean(flpvd, Weights(ws))
fpvd_warp = mean(flpvd_warp, Weights(ws))

println("Force (exact):       $(fhf + fpexact)")
println("Force (exact, warp): $(fhf_warp + fpexact_warp)")
println("Force (vd):          $(fhf + fpvd)")
println("Force (vd, warp):    $(fhf_warp + fpvd_warp)")

p1 = plot((flhf + flpvd)[2:end], reuse=false)
plot!((flhf_warp + flpvd_warp)[2:end])
display(p1)