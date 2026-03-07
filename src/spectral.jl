
"""
    normal_mode_decomposition(y, x; Lx, kmax=nothing)

y = f.(x) sampled on a uniform, periodic, half-open grid of length Lx.
Returns (a0, a, b) for cosine/sine Fourier series.

- "true_amplitude": uses 2/N scaling
- "true_phase": corrects for x not starting at 0 (e.g. x₁ = -Lx/2)
"""
function normal_mode_decomposition(y::AbstractVector, x::AbstractVector; Lx::Real, kmax=nothing)
    N = length(y)
    @assert length(x) == N
    dx = x[2] - x[1]
    @assert all(abs((x[i+1]-x[i]) - dx) < 1e-10 for i in 1:N-1) "x must be uniform"

    # FFT on nonnegative frequencies
    Y = rfft(y)

    # phase correction because FFT assumes samples at x = 0:dx:(N-1)dx
    x0 = x[1]
    ω0 = 2π / Lx
    for k in 0:length(Y)-1
        Y[k+1] *= exp(1im * (k*ω0) * x0)
    end

    # series coefficient scaling
    a0 = 2 * real(Y[1]) / N
    a  = 2 .* real.(Y[2:end]) ./ N
    b  = -2 .* imag.(Y[2:end]) ./ N

    if kmax !== nothing
        a = a[1:min(kmax, length(a))]
        b = b[1:min(kmax, length(b))]
    end
    return a0, a, b
end


"""
Precompute cosine basis matrix for modes m = m1:m2 using cos(2m*y).
Returns B of size (ny, m2-m1+1).
"""
function cosine_basis(y, m1, m2)
    ny = length(y)
    M  = m2 - m1 + 1
    B  = Matrix{Float64}(undef, ny, M)
    for (j, m) in enumerate(m1:m2)
        @inbounds B[:, j] .= cos.(2*m .* y)
    end
    return B
end

"""
Takes a shear flow returns the canonical parameters
q = 2ikPe and the even Fourier coefficients that describe the flow β_0, β_e, 
that are used to 
"""
function flow_decomposition(U, Pe, domain::Domain)
    y = domain.y
    Lx = domain.Ny
    K = domain.K
    qs = Complex{Float64}.(im .* big.(2*K*Pe)) # <--- make complex bigfloat?
    β0, β, βo = normal_mode_decomposition(U, y; Lx)

    # TODO
    # If flow not even, compute its even periodic extension and 
    # halve the domain (i.e. focus on a single half).

    return qs, β0, β
end

"""
collects all eigenpairs including eigenfunction sweeps for the entire range of
    alongstream wavenumber space.
"""
function spectral_basis(U, Pe, domain::Domain, Ns)
    qs, β0, β = flow_decomposition(U, Pe, domain)

    _, avals, Ak = collect_sweep_eigen(Even, qs, Ns, β)
    Phi_e = sweep_even_eigenfunctions(Ak, domain.y)

    _, bvals, Bk = collect_sweep_eigen(Odd, qs, Ns, β)
    Phi_o = sweep_odd_eigenfunctions(Bk, domain.y)

    return SpectralBasis(qs, β0, β, avals, bvals, Ak, Bk, Phi_e, Phi_o)
end