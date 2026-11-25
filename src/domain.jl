
"""
    domain(L, M, Nx, Ny)

Takes physical parameters and returns flow parameters used
for the computation of the eigenfuncions.

Grids:
- `x`: `Nx` points on the half-open periodic interval `[-Lx/2, Lx/2)`, `Lx = L / M`.
- `y`: `Ny` **cell-centred** points on `[0, πM]`, `y_j = (j - 1/2)Δy` with
  `Δy = πM / Ny`. Neither wall (`0`, `πM`) is a grid point. `.
"""
function domain(L::Real, M::Integer, Nx::Integer, Ny::Integer)
    Lx = L / M
    ω0 = 1 / Lx
    K = ω0 .* (0:(Nx ÷ 2))
    x = range(-Lx / 2, Lx / 2; length = Nx + 1)[1:(end - 1)]  # half-open periodic grid
    Δy = π * M / Ny
    # create y array with cell-centred points, walls not included
    y = ((1:Ny) .- 0.5) .* Δy
    return Domain(L, M, Nx, Ny, x, y, K)
end
