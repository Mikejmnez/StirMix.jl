module StirMix

# Write your package code here.
using ProgressMeter
using FFTW
using HillFunctions
using JLD2
using LinearAlgebra

include("types.jl")
include("domain.jl")
include("spectral_io.jl")
include("spectral.jl")
include("evolution.jl")
include("flow.jl")
include("reconstruction.jl")

export domain,
       rot_domain,
       spectral_basis,
       solve_explicit,
       evolve_explicit_avg

end
