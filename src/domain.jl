
"""
    domain(L, M, Nx, Ny)

Takes physical parameters and returns flow parameters used
for the computation of the eigenfuncions.

Grids:
- `x`: `Nx` **cell-centred** points on the periodic interval `[-Lx/2, Lx/2]`,
  `x_i = -Lx/2 + (i - 1/2)Δx` with `Δx = Lx / Nx`, `Lx = L / M`.
- `y`: `Ny` **cell-centred** points on `[0, πM]`, `y_j = (j - 1/2)Δy` with
  `Δy = πM / Ny`. Neither wall (`0`, `πM`) is a grid point.
"""
function domain(L::Real, M::Integer, Nx::Integer, Ny::Integer)
    Lx = L / M
    ω0 = 1 / Lx
    K = ω0 .* (0:(Nx ÷ 2))
    Δx = Lx / Nx
    # create x array with cell-centred points, consistent with y (ADR-002)
    x = -Lx / 2 .+ ((1:Nx) .- 0.5) .* Δx
    Δy = π * M / Ny
    # create y array with cell-centred points, walls not included
    y = ((1:Ny) .- 0.5) .* Δy
    return Domain(L, M, Nx, Ny, x, y, K)
end

"""
    even_extension(f, y) -> (f_ext, y_ext)

Even periodic extension of `f` (a vector, or a matrix with y along dim 1) from the
channel `[0, π]` to the period `[0, 2π)`: `f_ext = vcat(f, reverse(f; dims = 1))` on
`y_ext = vcat(y, π .+ y)`.

Exact (even about both walls, y = 0 and y = π) only on the cell-centred grid built by
`domain`; on a grid that samples y = 0 it would be even about −Δy/2 instead
(BUG-003). Assumes `M = 1`.
"""
function even_extension(f::AbstractVecOrMat, y::AbstractVector)
    size(f, 1) == length(y) ||
        throw(DimensionMismatch("size(f, 1) = $(size(f, 1)) but length(y) = $(length(y))"))
    return vcat(f, reverse(f; dims = 1)), vcat(y, π .+ y)
end

"""
    y_modes(f, dom; extend) -> (γ, γ_o)

Cosine coefficients `γ = [γ_0; γ_1; …]` and sine coefficients `γ_o` in y of `f`
(a vector, or a matrix with y along dim 1, decomposed column by column), truncated to
`Nγ` modes.

- `extend = true` (flows needing the even periodic extension): `f` is mirrored with
  [`even_extension`](@ref) and decomposed on `[0, 2π)`; `Nγ = Ny`.
- `extend = false`: `f` is decomposed as a π-periodic profile in the variable `2y`;
  `Nγ = Ny ÷ 2`.

This is the single y-decomposition used by `solve_explicit` and `evolve_explicit_avg`
(TASK-003), so the 2-D and the averaged paths always share one extension.
"""
function y_modes(f::AbstractVecOrMat, dom::Domain; extend::Bool)
    if extend
        f_ext, y_ext = even_extension(f, dom.y)
        χ_0, χ_e, γ_o = normal_mode_decomposition(f_ext, y_ext; Lx = 2π, dim = 1)
        Nγ = length(dom.y)
    else
        χ_0, χ_e, γ_o = normal_mode_decomposition(f, 2 .* dom.y; Lx = 2π, dim = 1)
        Nγ = length(dom.y) ÷ 2
    end
    if f isa AbstractVector
        return vcat(χ_0, χ_e)[1:Nγ], γ_o[1:Nγ]
    else
        return vcat(transpose(χ_0), χ_e)[1:Nγ, :], γ_o[1:Nγ, :]
    end
end
