struct WaveFunction{U, V, T}
    value::U
    gradient::V
    laplacian::T
end

mutable struct WaveFunctionStatus
    value::Float64
    gradient::Array{Float64}
    laplacian::Float64
end

function init_status(ψ, x)
    WaveFunctionStatus(
        ψ.value(x),
        ψ.gradient(x),
        ψ.laplacian(x)
    )
end