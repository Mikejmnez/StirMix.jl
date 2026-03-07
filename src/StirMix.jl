module StirMix

# Write your package code here.
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
    cosine_basis,
    reconstruct!,
    normal_mode_decomposition,
    spectral_basis

end
