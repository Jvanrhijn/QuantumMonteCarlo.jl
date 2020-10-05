struct Model{M1, M2, U, V, T}
    hamiltonian::M1
    hamiltonian_recompute::M2
    wave_function::WaveFunction{U, V, T}
end