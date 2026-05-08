module StirMix

# Write your package code here.
using ProgressMeter
using FFTW
using HillFunctions
using LinearAlgebra

include("types.jl")
include("domain.jl")
include("spectral.jl")
include("evolution.jl")
include("flow.jl")
include("reconstruction.jl")

export domain,
    flow_decomposition,
    reconstruct_fourier_series,
    cosine_tail_data,
    sine_tail_data,
    reconstruct!,
    normal_mode_decomposition,
    spectral_basis,
    complex_x_coeffs,
    reconstruct_from_complex_coeffs,
    solve_explicit,
    prepare_mode_data_avg,
    prepare_explicit_avg,
    evolve_explicit_avg,
    time_resample,
    predefined_flow

end
