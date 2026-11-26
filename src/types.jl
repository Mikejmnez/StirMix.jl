
struct Domain
    L::Float64
    M::Int
    Nx::Int
    Ny::Int
    x::Vector{Float64}
    y::Vector{Float64}
    K::Vector{Float64}
end

"""
    RotatedDomain

A shear flow rotated by 90° (flow along y, shear in x) on a box of integer aspect ratio
`L`, carried in two frames. Build it with [`rot_domain`](@ref).

- `phys`: the physical frame, `domain(L, 1, Nx, Ny)`.
- `solver`: the frame the solver runs in, `domain(1/L, 1, Ny, Nx)`.
- `L`: the aspect ratio of `phys`.
"""
struct RotatedDomain
    phys::Domain
    solver::Domain
    L::Int
end

struct SpectralBasis
    qs::Any
    β0::Float64
    β::Vector{Float64}
    an::Any
    bn::Any
    Ak::Any
    Bk::Any
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
    Tuple{Vector{SpectralBasis}, Vector{Float64}, Vector{Float64},
        Vector{Float64}, Vector{Int}, Vector{Tuple{Int, Int}}}
}
