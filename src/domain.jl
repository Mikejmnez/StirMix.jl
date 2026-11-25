
"""
    domain(L, M, Nx, Ny)

Takes physical parameters and returns flow parameters used
for the computation of the eigenfuncions.

Grids:
- `x`: `Nx` **cell-centred** points on the periodic interval `[-Lx/2, Lx/2]`,
  `x_i = -Lx/2 + (i - 1/2)Δx` with `Δx = Lx / Nx`, `Lx = L / M` (ADR-002).
- `y`: `Ny` **cell-centred** points on `[0, πM]`, `y_j = (j - 1/2)Δy` with
  `Δy = πM / Ny`. Neither wall (`0`, `πM`) is a grid point (ADR-001).
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
