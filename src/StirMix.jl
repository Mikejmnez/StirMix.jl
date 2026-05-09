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
    spectral_basis,
    solve_explicit,
    prepare_explicit_avg,
    evolve_explicit_avg

end
