
"""
    normal_mode_decomposition(y, x; Lx, kmax=nothing, dim=1)

`y` sampled on a uniform, periodic, half-open grid `x` of length `Lx`.
Returns `(a0, a, b)` for cosine/sine Fourier series along `dim`.

- "true_amplitude": uses 2/N scaling
- "true_phase": corrects for x not starting at 0 (e.g. x₁ = -Lx/2)

For vectors, `a0` is a scalar and `a`, `b` are vectors. For higher-rank
arrays, `a0` is the zero-mode slice with `dim` removed, while `a` and `b`
keep the input rank with the Fourier modes stored along `dim`.
"""
function normal_mode_decomposition(
        y::AbstractVector,
        x::AbstractVector;
        Lx::Real,
        kmax = nothing,
        dim::Integer = 1
)
    dim == 1 || throw(ArgumentError("dim must be 1 for vector inputs; got dim=$dim"))
    a0, a, b = _normal_mode_decomposition(y, x; Lx = Lx, kmax = kmax, dim = dim)
    return a0[], a, b
end

function normal_mode_decomposition(
        y::AbstractArray,
        x::AbstractVector;
        Lx::Real,
        kmax = nothing,
        dim::Integer = 1
)
    return _normal_mode_decomposition(y, x; Lx = Lx, kmax = kmax, dim = dim)
end

function _normal_mode_decomposition(
        y::AbstractArray,
        x::AbstractVector;
        Lx::Real,
        kmax = nothing,
        dim::Integer
)
    1 <= dim <= ndims(y) ||
        throw(ArgumentError("dim must be between 1 and $(ndims(y)); got dim=$dim"))

    N = size(y, dim)
    @assert length(x) == N
    @assert N>=2 "x must contain at least two points"
    dx = x[2] - x[1]
    @assert all(abs((x[i + 1] - x[i]) - dx) < 1e-10 for i in 1:(N - 1)) "x must be uniform"

    # phase correction because FFT assumes samples at x = 0:dx:(N-1)dx
    x0 = x[1]
    ω0 = 2π / Lx

    if eltype(y) <: Real
        # FFT on nonnegative frequencies
        Y = rfft(y, dim)
        nfreq = size(Y, dim)
        phase = reshape(
            [exp(-1im * (k * ω0) * x0) for k in 0:(nfreq - 1)],
            ntuple(d -> d == dim ? nfreq : 1, ndims(Y))
        )
        Y .*= phase

        # series coefficient scaling
        a0 = 2 .* real.(selectdim(Y, dim, 1)) ./ N
        a = 2 .* real.(selectdim(Y, dim, 2:nfreq)) ./ N
        b = -2 .* imag.(selectdim(Y, dim, 2:nfreq)) ./ N
    else
        # Complex-valued data need both +k and -k modes.
        Y = fft(y, dim)
        nfreq = N ÷ 2 + 1

        Cpos = copy(selectdim(Y, dim, 1:nfreq))
        phase_pos = reshape(
            [exp(-1im * (k * ω0) * x0) for k in 0:(nfreq - 1)],
            ntuple(d -> d == dim ? nfreq : 1, ndims(Y))
        )
        Cpos .*= phase_pos ./ N

        a0 = 2 .* selectdim(Cpos, dim, 1)

        neg_indices = [N - k + 1 for k in 1:(nfreq - 1)]
        Cneg = copy(selectdim(Y, dim, neg_indices))
        phase_neg = reshape(
            [exp(1im * (k * ω0) * x0) for k in 1:(nfreq - 1)],
            ntuple(d -> d == dim ? nfreq - 1 : 1, ndims(Y))
        )
        Cneg .*= phase_neg ./ N

        Cpos_modes = selectdim(Cpos, dim, 2:nfreq)
        a = Cpos_modes .+ Cneg
        b = 1im .* (Cpos_modes .- Cneg)
    end

    if kmax !== nothing
        kmax isa Integer || throw(ArgumentError("kmax must be an integer or nothing"))
        kmax >= 0 || throw(ArgumentError("kmax must be nonnegative"))
        nk = min(kmax, size(a, dim))
        a = copy(selectdim(a, dim, 1:nk))
        b = copy(selectdim(b, dim, 1:nk))
    end
    return a0, a, b
end

"""
    complex_x_coeffs(ce, co)

Convert cosine/sine Fourier coefficients in x into complex coefficients C
such that

    real(C[k] * exp(im* 2πK[k]*x)) = ce[k]*cos((2πK[k]*x) + co[k]*sin((2πK[k]*x)

Hence:
    C[k] = ce[k] - im*co[k]
"""
function complex_x_coeffs(ce::AbstractVector, co::AbstractVector)
    @assert length(ce)==length(co) "ce and co must have the same length"
    return complex.(ce, .-co)
end

"""
    complex_x_coeffs(ce, co)

Convert cosine/sine Fourier coefficients in x into complex coefficients C
such that

    real(C[k] * exp(im* 2πK[k]*x)) = ce[k]*cos((2πK[k]*x) + co[k]*sin((2πK[k]*x)

Hence:
    C[k] = ce[k] - im*co[k]
"""
function complex_x_coeffs(ce::AbstractMatrix, co::AbstractMatrix)
    @assert size(ce)==size(co) "ce and co must have the same shape"
    return complex.(ce, .-co)
end

# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────

"""Advance and wrap phi_old to [0, 2π]."""
function _update_phi_old(phi_old::Real, phi_new::Real)
    phi_old = phi_old + phi_new
    if phi_old > 2π
        phi_old -= 2π
    elseif phi_old < 0
        phi_old += 2π
    end
    return phi_old
end

"""
    coeff_project(phi, y; phi_old=π, phi_new=0.0)

Compute the even and odd Fourier coefficients of the 1-D signal `phi(y)` after
applying a phase shift `phi_new`.

# Arguments
- `phi`    : 1-D (complex) vector, `phi = phi(y)`
- `y`      : 1-D real vector of y in `[0, 2π]` coordinates (uniform spacing)
- `phi_old`: original centering of the flow (default π)
- `phi_new`: phase shift of the velocity field (default 0)

# Returns
- `even_coeffs`: complex vector of length `nL+1` — cosine coefficients `a_l`,
                 `l = 0, 1, …, nL`
- `odd_coeffs` : complex vector of length `nL`   — sine  coefficients `b_l`,
                 `l = 1, 2, …, nL`
- `phi_new`    : unchanged
- `phi_old`    : updated phase reference, wrapped to `[0, 2π]`

# Reconstruction
```
phi_shifted(y) ≈ Σ_{l=0}^{nL}  even_coeffs[l+1] * cos(l*y)
               + Σ_{l=1}^{nL}  odd_coeffs[l]    * sin(l*y)
```
"""
function coeff_project(
        phi::AbstractVector,
        y::AbstractVector;
        phi_old::Real = π,
        phi_new::Real = 0.0
)
    phi_old = _update_phi_old(phi_old, phi_new)
    L = length(y)
    nL = (L - 1) ÷ 2

    χc_0, χc, γc_o = normal_mode_decomposition(phi, y; Lx = 2π)

    phi_cos = [cos(-l * phi_new) for l in 1:nL]
    phi_sin = [sin(-l * phi_new) for l in 1:nL]

    e_coeffs = χc .* phi_cos .- γc_o .* phi_sin
    e_coeffs = vcat(χc_0, e_coeffs)  # prepend γc_0

    o_coeffs = γc_o .* phi_cos .+ χc .* phi_sin

    return e_coeffs, o_coeffs, phi_new, phi_old
end

# ──────────────────────────────────────────────────────────────────────────────
# 2-D method  (y along dimension 1, x along dimension 2)
# ──────────────────────────────────────────────────────────────────────────────

"""
    coeff_project(phi, y; phi_old=π, phi_new=0.0)

2-D overload: `phi` has shape `(Ly, Lx)` with the **y-dimension along dim 1**.
The FFT is applied along dim 1 only; the x-dimension is carried through.

# Returns
- `even_coeffs`: complex matrix of shape `(nL+1, Lx)` — row index is wavenumber `l`
- `odd_coeffs` : complex matrix of shape `(nL,   Lx)` — row index is wavenumber `l`
- `phi_new`, `phi_old`: same semantics as the 1-D method

# Reconstruction (at a fixed x-column `j`)
```
phi_shifted(y) ≈ Σ_{l=0}^{nL}  even_coeffs[l+1, j] * cos(l*y)
               + Σ_{l=1}^{nL}  odd_coeffs[l,   j] * sin(l*y)
```
"""
function coeff_project(
        phi::AbstractMatrix,
        y::AbstractVector;
        phi_old::Real = π,
        phi_new::Real = 0.0
)
    phi_old = _update_phi_old(phi_old, phi_new)

    L = length(y)
    nL = (L - 1) ÷ 2
    norm = (y[end] - y[1]) / (2 * (y[2] - y[1]))

    # FFT along dim 1 (the y-axis).  F[k+1, :] ↔ wavenumber k.
    F = fft(phi, 1)

    F_pos = F[1:(nL + 1), :]
    F_neg_rev = [F[1:1, :]; F[L:-1:(L - nL + 1), :]]   # (nL+1, Lx)

    e_coeffs = 0.5 .* (F_pos .+ F_neg_rev) ./ norm
    e_coeffs[1, :] ./= 2
    o_coeffs = 0.5 .* (F_pos .- F_neg_rev) ./ (-1im * norm)

    # Column vector (nL+1, 1) so it broadcasts against (nL+1, Lx)
    phi_cos = reshape([cos(-l * phi_new) for l in 0:nL], :, 1)
    phi_sin = reshape([sin(-l * phi_new) for l in 0:nL], :, 1)

    da_even = e_coeffs .* phi_cos .- o_coeffs .* phi_sin   # (nL+1, Lx)
    da_odd = o_coeffs .* phi_cos .+ e_coeffs .* phi_sin   # (nL+1, Lx)

    return da_even, da_odd[2:end, :], phi_new, phi_old
end

"""
    cosine_tail_data(y, γ, Nn)

Build the cosine basis and coefficient views for the tail modes m = Nn:Mmax,
where Mmax = length(γ) - 1.

Returns `(m_tail, Atail, γtail, tmp)` where:

- `m_tail` : vector of mode indices Nn:Mmax
- `Atail`  : cosine basis matrix with columns cos.(2m*y)
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

        Atail = Matrix{Float64}(undef, ny, Mt)
        for (j, m) in enumerate(m_tail)
            @inbounds Atail[:, j] .= cos.(2m .* y)
        end

        γtail = @view γ[(Nn + 1):(Mmax + 1)]           # γ[mode m] stored at index m+1
        tmp = similar(γtail)
    else
        m_tail = Int[]
        Atail = zeros(Float64, length(y), 0)
        γtail = @view γ[1:0]
        tmp = similar(γtail)
    end

    return m_tail, Atail, γtail, tmp
end

"""
    sine_tail_data(y, γ, Nn)

Build the sine basis and coefficient views for the tail modes m = Nn+1:Mmax,
where Mmax = length(γ).

Returns `(m_tail, Btail, γtail, tmp)` where:

- `m_tail` : vector of mode indices Nn+1:Mmax
- `Btail`  : sine basis matrix with columns sin.(2m*y)
- `γtail`  : view into γ corresponding to modes Nn+1:Mmax
- `tmp`    : work array similar to γtail

If `Mmax < Nn + 1`, returns empty objects of compatible size.
"""
function sine_tail_data(y::AbstractVector, γ::AbstractVector, Nn::Integer)
    Mmax = length(γ)

    if Mmax >= Nn + 1
        m_tail = collect((Nn + 1):Mmax)            # modes m = Nn+1, ..., Mmax
        Mt = length(m_tail)
        ny = length(y)

        Btail = Matrix{Float64}(undef, ny, Mt)
        for (j, m) in enumerate(m_tail)
            @inbounds Btail[:, j] .= sin.(2m .* y)
        end

        γtail = @view γ[(Nn + 1):Mmax]               # γ[mode m] stored at index m
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
        precision_policy = :mixed,
        qc0::Union{Nothing, Real} = nothing,
        qc::Union{Nothing, Real} = nothing)
    absq = abs.(2 .* (2π * K) .* Pe)
    if precision_policy === :mixed
        qc0 === nothing &&
            throw(ArgumentError("qc0 is required when precision_policy=:mixed"))
        qc === nothing &&
            throw(ArgumentError("qc is required when precision_policy=:mixed"))
        qc0 < qc || throw(ArgumentError("qc0 must be less than qc; got qc0=$qc0, qc=$qc"))

        idx64 = findall(absq .< qc0)
        idxbig = findall((absq .>= qc0) .& (absq .< qc))
        idxHR = findall(absq .>= qc)

        qs64 = ComplexF64.(complex.(0.0, 2 .* (2π * K[idx64]) .* Pe))

        qsbig = complex.(zero(BigFloat), big(4π) .* BigFloat.(K[idxbig]) .* BigFloat(Pe))

        qsHR = ComplexF64.(complex.(0.0, 2 .* (2π * K[idxHR]) .* Pe))

        return (; idx64, idxbig, idxHR, qs64, qsbig, qsHR)

    elseif precision_policy === :float64
        idx64 = collect(eachindex(K))
        qs64 = ComplexF64.(complex.(0.0, 2 .* (2π * K) .* Pe))

    else
        throw(ArgumentError(
            "precision_policy must be one of (:mixed, :float64); got $(repr(precision_policy))"
        ))
    end

    return (; idx64, idxbig = Int[], idxHR = Int[], qs64,
        qsbig = Complex{BigFloat}[], qsHR = ComplexF64[])
end

"""
Takes a shear flow and returns its sampled values plus the even Fourier
coefficients β_0 and β_e.
"""
function flow_decomposition(U::AbstractVector, domain::Domain)
    even_extension = false
    y = domain.y
    Ny = 2 * domain.Ny
    β0, β, β_o = normal_mode_decomposition(U, y; Lx = Ny)
    nb = length(β_o) ÷ 2
    if any(abs.(β_o[1:nb]) .> 1e-2)
        even_extension = true
        U1 = U .- (β0 / 2)
        U2 = -U1
        U = (β0 / 2) .+ vcat(U1, U2)
        ny = vcat(y, pi .+ y)
        β0, β, _ = normal_mode_decomposition(U, ny; Lx = Ny)
    end
    return U, β0, β, even_extension
end

function flow_decomposition(flow::Symbol, domain::Domain)
    U, β0, β, even_extension = predefined_flow(flow, domain; Nmodes = length(domain.y) ÷ 2)

    return U, β0, β, even_extension
end

function _scatter_spectral_point!(qs, an, bn, Ak, Bk,
        k, q, Ns, β;
        Nmax = nothing,
        compute_odd::Bool = true
)
    λe, Ve = adaptive_eigen(Even, q, Ns, β; Nmax = Nmax)

    qs[k] = q
    an[k] = λe
    Ak[k] = Ve

    if compute_odd
        λo, Vo = adaptive_eigen(Odd, q, Ns, β; Nmax = Nmax)

        bn[k] = λo
        Bk[k] = Vo
    end

    return nothing
end

function _scatter_spectral_batch!(qs, an, bn, Ak, Bk,
        idx, qbatch, Ns, β;
        Nmax = nothing,
        advance_progress = nothing,
        compute_odd::Bool = true,
        parallel::Bool = false
)
    isempty(idx) && return nothing

    units_per_q = 1 + Int(compute_odd)
    if parallel && Threads.nthreads() > 1 && length(qbatch) > 1
        Threads.@threads :dynamic for j in 1:length(qbatch)
            k = idx[j]
            q = qbatch[j]

            _scatter_spectral_point!(qs, an, bn, Ak, Bk,
                k, q, Ns, β;
                Nmax = Nmax,
                compute_odd = compute_odd)

            advance_progress === nothing || advance_progress(units_per_q)
        end
    else
        for j in 1:length(qbatch)
            k = idx[j]
            q = qbatch[j]

            _scatter_spectral_point!(qs, an, bn, Ak, Bk,
                k, q, Ns, β;
                Nmax = Nmax,
                compute_odd = compute_odd)

            advance_progress === nothing || advance_progress(units_per_q)
        end
    end

    return nothing
end

function _normalize_sweep_chunk_size(sweep_chunk_size::Union{Nothing, Integer})
    sweep_chunk_size === nothing && return nothing
    sweep_chunk_size > 0 ||
        throw(ArgumentError("sweep_chunk_size must be positive"))
    return Int(sweep_chunk_size)
end

function _scatter_spectral_group!(qs, an, bn, Ak, Bk,
        idx::AbstractVector, qvals::AbstractVector,
        Ns, β;
        sweep_chunk_size::Union{Nothing, Int} = nothing,
        parallel::Bool = false,
        kwargs...)
    isempty(idx) && return nothing
    length(idx) == length(qvals) ||
        throw(ArgumentError("idx and qvals must have the same length"))

    if sweep_chunk_size === nothing
        _scatter_spectral_batch!(qs, an, bn, Ak, Bk,
            idx, qvals, Ns, β;
            parallel = parallel,
            kwargs...)
        return nothing
    end

    n = length(idx)
    for first in 1:sweep_chunk_size:n
        last = min(first + sweep_chunk_size - 1, n)

        idx_chunk = @view idx[first:last]
        q_chunk = @view qvals[first:last]

        _scatter_spectral_batch!(qs, an, bn, Ak, Bk,
            idx_chunk, q_chunk, Ns, β;
            parallel = parallel,
            kwargs...)
    end

    return nothing
end

function _collect_spectral_data(domain::Domain, Pe::Real, Ns, β;
        Nmax = nothing,
        precision_policy = :float64,
        parallel::Bool = true,
        show_progress::Bool = true,
        qc0::Union{Nothing, Real} = nothing,
        qc::Union{Nothing, Real} = nothing,
        compute_odd::Bool = true,
        sweep_chunk_size::Union{Nothing, Integer} = nothing,
        even_extension::Bool = false)
    sweep_chunk_size = _normalize_sweep_chunk_size(sweep_chunk_size)

    fac = even_extension ? 2 : 1.0

    nk = length(domain.K)
    qs = Vector{Union{ComplexF64, Complex{BigFloat}}}(undef, nk)

    an = Vector{Any}(undef, nk)
    Ak = Vector{Any}(undef, nk)

    bn = compute_odd ? Vector{Any}(undef, nk) : nothing
    Bk = compute_odd ? Vector{Any}(undef, nk) : nothing

    qdata = _q_values(
        fac .* domain.K, Pe; precision_policy = precision_policy, qc0 = qc0, qc = qc)

    # create one progress bar here
    nq = nk
    @assert nq == length(qdata.idx64) + length(qdata.idxbig) + length(qdata.idxHR)

    eigenpair_passes = 1 + Int(compute_odd)
    total = nq * (eigenpair_passes)

    # define one local advance
    p = show_progress && total > 0 ?
        Progress(total; desc = "Computing spectral basis") :
        nothing

    processed = Ref(0)
    progress_lock = ReentrantLock()

    advance_progress = p === nothing ? nothing : n -> begin
        lock(progress_lock)
        try
            processed[] += n
            update!(p, processed[])
        finally
            unlock(progress_lock)
        end
        return nothing
    end

    completed = false

    try
        _scatter_spectral_group!(qs, an, bn, Ak, Bk,
            qdata.idx64, qdata.qs64, Ns, β;
            Nmax = Nmax,
            parallel = parallel,
            advance_progress = advance_progress,
            compute_odd = compute_odd,
            sweep_chunk_size = sweep_chunk_size)

        _scatter_spectral_group!(qs, an, bn, Ak, Bk,
            qdata.idxbig, qdata.qsbig, Ns, β;
            Nmax = Nmax,
            parallel = parallel,
            advance_progress = advance_progress,
            compute_odd = compute_odd,
            sweep_chunk_size = sweep_chunk_size)

        _scatter_spectral_group!(qs, an, bn, Ak, Bk,
            qdata.idxHR, qdata.qsHR, Ns, β;
            Nmax = Nmax,
            advance_progress = advance_progress,
            parallel = parallel,
            compute_odd = compute_odd,
            sweep_chunk_size = sweep_chunk_size)
        completed = true

    finally
        if p !== nothing
            if completed
                finish!(p)
            else
                cancel(p, "Spectral basis computation aborted")
            end
        end
    end
    return (; qs, an, bn, Ak, Bk)
end

"""
    spectral_basis(U, Pe, domain, Ns;
                   precision_policy=:float64,
                   qc0=1000.0,
                   qc=nothing,
                   compute_odd=false,
                   parallel=true,
                   show_progress=true,
                   sweep_chunk_size=nothing,
                   filename=nothing,
                   spectral_io=:compute)

Collect eigenpairs for the alongstream wavenumber sweep.

When `filename` is provided, `spectral_io` controls local spectral-data I/O:
`:compute` ignores disk, `:read` returns data from `filename`, and `:write` or
`:refresh` compute the basis and then write it to `filename`. Phase 1 read mode
checks only that the expected datasets exist; it does not validate stored
metadata against the current arguments.

"""
function spectral_basis(U::AbstractVector, Pe, domain::Domain, Ns;
        precision_policy = :float64, qc0::Real = 1000.0,
        qc::Union{Nothing, Real} = nothing,
        compute_odd::Bool = false,
        parallel::Bool = true,
        show_progress::Bool = true,
        sweep_chunk_size::Union{Nothing, Integer} = nothing,
        filename = nothing,
        spectral_io = :compute)
    spectral_path, spectral_mode = _normalize_spectral_io(filename, spectral_io)
    spectral_mode === :read && return _read_spectral_data(spectral_path)

    _, β0, β, even_extension = flow_decomposition(U, domain)
    if even_extension
        β0 *= 2
    end
    data = _collect_spectral_data(domain, abs(Pe), Ns, β;
        precision_policy = precision_policy, qc0 = qc0, qc = qc,
        parallel = parallel,
        show_progress = show_progress,
        compute_odd = compute_odd,
        sweep_chunk_size = sweep_chunk_size,
        even_extension = even_extension)

    basis = SpectralBasis(
        data.qs, β0, β, data.an, data.bn, data.Ak, data.Bk, Pe, even_extension)
    if spectral_mode in (:write, :refresh)
        meta = _spectral_io_metadata(domain, Pe, Ns, β0, β;
            precision_policy = precision_policy,
            qc0 = qc0,
            qc = qc,
            compute_odd = compute_odd,
            even_extension = even_extension)
        _write_spectral_data(spectral_path, basis, meta)
    end

    return basis
end

"""
    spectral_basis(flow, Pe, domain, Ns=nothing;
                   precision_policy=:float64,
                   qc0=1000.0,
                   qc=nothing,
                   compute_odd=false,
                   parallel=true,
                   show_progress=true,
                   sweep_chunk_size=nothing,
                   filename=nothing,
                   spectral_io=:compute)

Collect eigenpairs for a predefined flow over the alongstream wavenumber sweep.

`filename` and `spectral_io` have the same behavior as the vector-flow method:
`:compute` ignores disk, `:read` reconstructs a basis from local spectral data,
and `:write`/`:refresh` write the computed basis after the existing adaptive
eigensolve path completes.
"""
function spectral_basis(
        flow::Symbol, Pe, domain::Domain, Ns::Union{Nothing, Integer} = nothing;
        precision_policy = :float64, qc0::Real = 1000.0,
        qc::Union{Nothing, Real} = nothing,
        compute_odd::Bool = false,
        parallel::Bool = true,
        show_progress::Bool = true,
        sweep_chunk_size::Union{Nothing, Integer} = nothing,
        filename = nothing,
        spectral_io = :compute)
    spectral_path, spectral_mode = _normalize_spectral_io(filename, spectral_io)
    spectral_mode === :read && return _read_spectral_data(spectral_path)

    _, β0, β, even_extension = flow_decomposition(flow, domain)
    if even_extension
        compute_odd = true
        β0 *= 2
    end

    data = _collect_spectral_data(domain, abs(Pe), Ns, β;
        precision_policy = precision_policy, qc0 = qc0, qc = qc,
        parallel = parallel,
        show_progress = show_progress,
        compute_odd = compute_odd,
        sweep_chunk_size = sweep_chunk_size,
        even_extension = even_extension)

    basis = SpectralBasis(
        data.qs, β0, β, data.an, data.bn, data.Ak, data.Bk, Pe, even_extension)
    if spectral_mode in (:write, :refresh)
        meta = _spectral_io_metadata(domain, Pe, Ns, β0, β;
            precision_policy = precision_policy,
            qc0 = qc0,
            qc = qc,
            compute_odd = compute_odd,
            even_extension = even_extension)
        _write_spectral_data(spectral_path, basis, meta)
    end

    return basis
end

function compute_bases(U::Union{Symbol, AbstractVector}, dom::Domain, Pe::Real,
        ft::Vector{Float64}, Ns::Integer; nt::Integer = 1)::BasesIter
    nft, _, _, vals_pos, ivals_pos = StirMix.time_resample(ft; nt = nt)
    Pe_set = Pe * vals_pos
    bases = [spectral_basis(U, Pe, dom, Ns) for Pe in Pe_set]
    Pe_vals = Pe * nft # truly negative values
    indT = StirMix.indt_intervals(ivals_pos)

    return (basis = bases, nft = nft, Pes = Pe_vals,
        vals = vals_pos, ivals = ivals_pos, indT = indT)
end
