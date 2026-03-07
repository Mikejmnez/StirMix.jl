
"""
domain()

Takes physical parameters and returns flow parameters used  
for the computation of the eigenfuncions

"""
function domain(L::Real, M::Integer, Nx::Integer, Ny::Integer)
    Lx = 2* L * π / M
    ω0 = 2π / Lx
    K = ω0 .* (0:Nx÷2)
    x  = range(-Lx/2, Lx/2; length=Nx+1)[1:end-1]  # half-open periodic grid
    y = range(0, π*M; length=(Ny ÷ 2))[1:end-1] # notice the lack of end point
    return Domain(L, M, Nx, Ny, x, y, K)

end
