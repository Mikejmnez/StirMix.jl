
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
    β0::Float64
    β::Vector{Float64}
    an
    bn
    Ak
    Bk
    Pe::Float64
    even_extension::Bool
end

struct AvgModeWork{T}
    A0::Vector{T}
    λ::Vector{T}
    γproj::Vector{T}
    e::Vector{T}
end

const BasesIter = NamedTuple{
    (:basis, :nft, :Pes, :vals, :ivals, :indT),
    Tuple{Vector{SpectralBasis},Vector{Float64},Vector{Float64},Vector{Float64},Vector{Int},Vector{Tuple{Int,Int}}},
}
