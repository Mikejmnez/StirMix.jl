
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
    complex_x_coeffs(ce, co)

Convert cosine/sine Fourier coefficients in x into complex coefficients C
such that

    real(C[k] * exp(im*K[k]*x)) = ce[k]*cos(K[k]*x) + co[k]*sin(K[k]*x)

Hence:
    C[k] = ce[k] - im*co[k]
"""
function complex_x_coeffs(ce::AbstractVector, co::AbstractVector)
    @assert length(ce) == length(co) "ce and co must have the same length"
    return complex.(ce, .-co)
end


"""
    cosine_tail_data(y, γ, Nn)

Build the cosine basis and coefficient views for the tail modes m = Nn:Mmax,
where Mmax = length(γ) - 1.

Returns `(m_tail, Btail, γtail, tmp)` where:

- `m_tail` : vector of mode indices Nn:Mmax
- `Btail`  : cosine basis matrix with columns cos.(2m*y)
- `γtail`  : view into γ corresponding to modes Nn:Mmax
- `tmp`    : work array similar to γtail

If `Mmax < Nn`, returns empty objects of compatible size.
"""
function cosine_tail_data(y::AbstractVector, γ::AbstractVector, Nn::Integer)
    Mmax = length(γ) - 1

    if Mmax >= Nn
        m_tail = collect(Nn:Mmax)                  # modes m = Nn, ..., Mmax
        Mt = length(m_tail)
        ny = length(y)

        Btail = Matrix{Float64}(undef, ny, Mt)
        for (j, m) in enumerate(m_tail)
            @inbounds Btail[:, j] .= cos.(2m .* y)
        end

        γtail = @view γ[(Nn+1):(Mmax+1)]           # γ[mode m] stored at index m+1
        tmp = similar(γtail)
    else
        m_tail = Int[]
        Btail = zeros(Float64, length(y), 0)
        γtail = @view γ[1:0]
        tmp = similar(γtail)
    end

    return m_tail, Btail, γtail, tmp
end


function _q_values(K::AbstractVector, Pe::Real; 
                  precision_policy=:mixed, qc1::Real=10000, qc2::Real=30000)
    absq = abs.(2 .* K .* Pe)
    if precision_policy === :mixed
        qc1 < qc2 || throw(ArgumentError("qc1 must be less than qc2; got qc1=$qc1, qc2=$qc2"))
        idx64  = findall(absq .< qc1)
        idxbig     = findall((absq .>= qc1) .& (absq .< qc2))
        idxHR = findall(absq .>= qc2)

        qs64 = ComplexF64.(complex.(0.0, 2 .* K[idx64] .* Pe))

        qsbig = complex.(
            zero(BigFloat),
            big(2) .* BigFloat.(K[idxbig]) .* BigFloat(Pe),
        )
        qsHR = ComplexF64.(complex.(0.0, 2 .* K[idxHR] .* Pe))

        return (; idx64, idxbig, idxHR, qs64, qsbig, qsHR)

    elseif precision_policy === :float64

        idx64 = collect(eachindex(K))
        qs64 = ComplexF64.(complex.(0.0, 2 .* K .* Pe))

    else
        throw(ArgumentError(
            "precision_policy must be one of (:mixed, :float64); got $(repr(precision_policy))"
        ))
    end

    return (; idx64, idxbig=Int[], idxHR=Int[], qs64, qsbig=Complex{BigFloat}[], qsHR=ComplexF64[])
end


"""
Takes a shear flow and returns its sampled values plus the even Fourier
coefficients β_0 and β_e.
"""
function flow_decomposition(U::AbstractVector, Pe, domain::Domain)
    y = domain.y
    Lx = domain.Ny
    K = domain.K
    β0, β, _ = normal_mode_decomposition(U, y; Lx)

    # TODO
    # If flow not even, compute its even periodic extension and 
    # halve the domain (i.e. focus on a single half).

    return U, β0, β
end


function flow_decomposition(flow::Symbol, Pe, domain::Domain)
    K = domain.K

    U, β0, β = predefined_flow(flow, domain; Nmodes=length(K))

    return U, β0, β
end


function _scatter_spectral_batch!(qs, an, bn, Ak, Bk, Φe, Φo,
                                  idx, qbatch, domain, Ns, β;
                                  Nmax=nothing,
                                  compute_eigenfunctions::Bool=true,
                                  compute_odd::Bool=true)
    isempty(idx) && return nothing

    qvec, avals, Ak_batch = collect_sweep_eigen(Even, qbatch, Ns, β; Nmax=Nmax)
    Φe_batch = compute_eigenfunctions ? sweep_even_eigenfunctions(Ak_batch, domain.y) : nothing

    bvals = Bk_batch = Φo_batch = nothing
    if compute_odd
        _, bvals, Bk_batch = collect_sweep_eigen(Odd, qbatch, Ns, β; Nmax=Nmax)
        Φo_batch = compute_eigenfunctions ? sweep_odd_eigenfunctions(Bk_batch, domain.y) : nothing
    end

    for (j, k) in pairs(idx)
        qs[k] = qvec[j]
        an[k] = avals[j]
        Ak[k] = Ak_batch[j]

        if compute_eigenfunctions
            Φe[k] = Φe_batch[j]
        end

        if compute_odd
            bn[k] = bvals[j]
            Bk[k] = Bk_batch[j]

            if compute_eigenfunctions
                Φo[k] = Φo_batch[j]
            end
        end
    end

    return nothing
end


function _collect_spectral_data(domain::Domain, Pe::Real, Ns, β;
                                Nmax=nothing,
                                precision_policy=:mixed,
                                qc1=10000, qc2=30000,
                                compute_eigenfunctions::Bool=true,
                                compute_odd::Bool=true)
    nk = length(domain.K)
    qs = Vector{Union{ComplexF64, Complex{BigFloat}}}(undef, nk)

    an = Vector{Any}(undef, nk)
    Ak = Vector{Any}(undef, nk)
    Φe = compute_eigenfunctions ? Vector{Any}(undef, nk) : nothing

    bn = compute_odd ? Vector{Any}(undef, nk) : nothing
    Bk = compute_odd ? Vector{Any}(undef, nk) : nothing
    Φo = (compute_odd && compute_eigenfunctions) ? Vector{Any}(undef, nk) : nothing

    qdata = _q_values(domain.K, Pe; precision_policy=precision_policy, qc1=qc1, qc2=qc2)

    _scatter_spectral_batch!(qs, an, bn, Ak, Bk, Φe, Φo,
                             qdata.idx64, qdata.qs64, domain, Ns, β;
                             Nmax=Nmax,
                             compute_eigenfunctions=compute_eigenfunctions,
                             compute_odd=compute_odd)

    _scatter_spectral_batch!(qs, an, bn, Ak, Bk, Φe, Φo,
                             qdata.idxbig, qdata.qsbig, domain, Ns, β;
                             Nmax=Nmax,
                             compute_eigenfunctions=compute_eigenfunctions,
                             compute_odd=compute_odd)

    _scatter_spectral_batch!(qs, an, bn, Ak, Bk, Φe, Φo,
                             qdata.idxHR, qdata.qsHR, domain, Ns, β;
                             Nmax=Nmax,
                             compute_eigenfunctions=compute_eigenfunctions,
                             compute_odd=compute_odd)

    return (; qs, an, bn, Ak, Bk, Φe, Φo)
end


"""
Collect eigenpairs for the alongstream wavenumber sweep.

If `compute_eigenfunctions=false`, the eigenvector/eigenvalue data are still
returned but `Φe` and `Φo` are set to `nothing`.
"""
function spectral_basis(U::AbstractVector, Pe, domain::Domain, Ns;
                        precision_policy=:mixed, qc1=10000, qc2=30000,
                        compute_eigenfunctions::Bool=true)
    _, β0, β = flow_decomposition(U, Pe, domain)
    data = _collect_spectral_data(domain, Pe, Ns, β;
                                  precision_policy=precision_policy, qc1=qc1, qc2=qc2,
                                  compute_eigenfunctions=compute_eigenfunctions)

    return SpectralBasis(data.qs, β0, β, data.an, data.bn, data.Ak, data.Bk,
                         data.Φe, data.Φo)
end


"""
Collect eigenpairs for a predefined flow over the alongstream wavenumber sweep.

If `compute_eigenfunctions=false`, the eigenvector/eigenvalue data are still
returned but `Φe` and `Φo` are set to `nothing`.
"""
function spectral_basis(flow::Symbol, Pe, domain::Domain, Ns;
                        precision_policy=:mixed, qc1=10000, qc2=30000,
                        compute_eigenfunctions::Bool=true)
    _, β0, β = flow_decomposition(flow, Pe, domain)
    data = _collect_spectral_data(domain, Pe, Ns, β;
                                  precision_policy=precision_policy, qc1=qc1, qc2=qc2, 
                                  compute_eigenfunctions=compute_eigenfunctions)

    return SpectralBasis(data.qs, β0, β, data.an, data.bn, data.Ak, data.Bk,
                         data.Φe, data.Φo)
end
