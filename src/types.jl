
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
    an
    bn
    Ak
    Bk
end

struct AvgModeWork{T}
    A0::Vector{T}
    λ::Vector{T}
    γproj::Vector{T}
    e::Vector{T}
end
