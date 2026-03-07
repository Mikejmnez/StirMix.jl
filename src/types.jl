
struct Domain
    L::Float64
    M::Int
    Nx::Int
    Ny::Int
    x::Vector{Float64}
    y::Vector{Float64}
    K::Vector{Float64}
end


struct SpectralBasis
    qs
    β0
    β
    avals
    bvals
    Ak
    Bk
    Phi_e
    Phi_o
end
