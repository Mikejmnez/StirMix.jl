
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

    real(C[k] * exp(im* 2πK[k]*x)) = ce[k]*cos((2πK[k]*x) + co[k]*sin((2πK[k]*x)

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

        γtail = @view γ[(Nn+1):(Mmax+1)]           # γ[mode m] stored at index m+1
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

        γtail = @view γ[(Nn+1):Mmax]               # γ[mode m] stored at index m
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
                  precision_policy=:mixed, 
                  qc1::Union{Nothing,Real}=nothing, 
                  qc2::Union{Nothing,Real}=nothing)
    absq = abs.(2 .* (2π * K) .* Pe)
    if precision_policy === :mixed
        qc1 < qc2 || throw(ArgumentError("qc1 must be less than qc2; got qc1=$qc1, qc2=$qc2"))
        idx64  = findall(absq .< qc1)
        idxbig     = findall((absq .>= qc1) .& (absq .< qc2))
        idxHR = findall(absq .>= qc2)

        qs64 = ComplexF64.(complex.(0.0, 2 .* (2π * K[idx64]) .* Pe))

        qsbig = complex.(zero(BigFloat), big(4π) .* BigFloat.(K[idxbig]) .* BigFloat(Pe))

        qsHR = ComplexF64.(complex.(0.0, 2 .* (2π * K[idxHR]) .* Pe))

        return (; idx64, idxbig, idxHR, qs64, qsbig, qsHR)

    elseif precision_policy === :float64

        idx64 = collect(eachindex(K))
        qs64 = ComplexF64.(complex.(0.0, 2 .* (2π * K).* Pe))

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
function flow_decomposition(U::AbstractVector, domain::Domain)
    y = domain.y
    Ny = 2 * domain.Ny
    β0, β, β_o = normal_mode_decomposition(U, y; Lx=Ny)
    nb = length(β_o)÷2
    if any(abs.(β_o[1:nb]) .> 1e-2)
        U1 = U .- (β0/2)
        U2 = - U1
        U = (β0/2) .+ vcat(U1, U2)
        ny = vcat(y, pi .+ y)
        β0, β, _ = normal_mode_decomposition(U, ny; Lx=Ny)
    end 
    return U, β0, β
end


function flow_decomposition(flow::Symbol, domain::Domain)

    U, β0, β = predefined_flow(flow, domain; Nmodes=length(domain.y)÷2)

    return U, β0, β
end


function _scatter_spectral_point!(qs, an, bn, Ak, Bk,
                                  k, q, domain, Ns, β;
                                  Nmax=nothing,
                                  compute_odd::Bool=true)
    λe, Ve = adaptive_eigen(Even, q, Ns, β; Nmax=Nmax)

    qs[k] = q
    an[k] = λe
    Ak[k] = Ve

    if compute_odd
        λo, Vo = adaptive_eigen(Odd, q, Ns, β; Nmax=Nmax)

        bn[k] = λo
        Bk[k] = Vo

    end

    return nothing
end


function _scatter_spectral_batch!(qs, an, bn, Ak, Bk,
                                  idx, qbatch, domain, Ns, β;
                                  Nmax=nothing,
                                  advance_progress=nothing,
                                  compute_odd::Bool=true,
                                  parallel::Bool=false)
    isempty(idx) && return nothing

    units_per_q = 1 + Int(compute_odd)
    if parallel && Threads.nthreads() > 1 && length(qbatch) > 1
        Threads.@threads :dynamic for j in 1:length(qbatch)
            k = idx[j]
            q = qbatch[j]

            _scatter_spectral_point!(qs, an, bn, Ak, Bk,
                                     k, q, domain, Ns, β;
                                     Nmax=Nmax,
                                     compute_odd=compute_odd)

            advance_progress === nothing || advance_progress(units_per_q)
        end
    else
        for j in 1:length(qbatch)
            k = idx[j]
            q = qbatch[j]

            _scatter_spectral_point!(qs, an, bn, Ak, Bk,
                                     k, q, domain, Ns, β;
                                     Nmax=Nmax,
                                     compute_odd=compute_odd)

            advance_progress === nothing || advance_progress(units_per_q)
        end
    end

    return nothing
end


function _normalize_sweep_chunk_size(sweep_chunk_size::Union{Nothing,Integer})
    sweep_chunk_size === nothing && return nothing
    sweep_chunk_size > 0 ||
        throw(ArgumentError("sweep_chunk_size must be positive"))
    return Int(sweep_chunk_size)
end


function _scatter_spectral_group!(qs, an, bn, Ak, Bk,
                                  idx::AbstractVector, qvals::AbstractVector,
                                  domain, Ns, β;
                                  sweep_chunk_size::Union{Nothing,Int}=nothing,
                                  parallel::Bool=false,
                                  kwargs...)
    isempty(idx) && return nothing
    length(idx) == length(qvals) ||
        throw(ArgumentError("idx and qvals must have the same length"))

    if sweep_chunk_size === nothing
        _scatter_spectral_batch!(qs, an, bn, Ak, Bk,
                                 idx, qvals, domain, Ns, β;
                                 parallel=parallel,
                                 kwargs...)
        return nothing
    end

    n = length(idx)
    for first in 1:sweep_chunk_size:n
        last = min(first + sweep_chunk_size - 1, n)

        idx_chunk = @view idx[first:last]
        q_chunk = @view qvals[first:last]

        _scatter_spectral_batch!(qs, an, bn, Ak, Bk,
                                 idx_chunk, q_chunk, domain, Ns, β;
                                 parallel=parallel,
                                 kwargs...)
    end

    return nothing
end


function _collect_spectral_data(domain::Domain, Pe::Real, Ns, β;
                                Nmax=nothing,
                                precision_policy=:float64,
                                parallel::Bool=true,
                                show_progress::Bool=true,
                                qc1::Union{Nothing,Real}=nothing, 
                                qc2::Union{Nothing,Real}=nothing,
                                compute_odd::Bool=true, 
                                sweep_chunk_size::Union{Nothing,Integer}=nothing)
    sweep_chunk_size = _normalize_sweep_chunk_size(sweep_chunk_size)

    nk = length(domain.K)
    qs = Vector{Union{ComplexF64, Complex{BigFloat}}}(undef, nk)

    an = Vector{Any}(undef, nk)
    Ak = Vector{Any}(undef, nk)


    bn = compute_odd ? Vector{Any}(undef, nk) : nothing
    Bk = compute_odd ? Vector{Any}(undef, nk) : nothing

    qdata = _q_values(domain.K, Pe; precision_policy=precision_policy, qc1=qc1, qc2=qc2)

    # create one progress bar here
    nq = nk
    @assert nq == length(qdata.idx64) + length(qdata.idxbig) + length(qdata.idxHR)

    eigenpair_passes = 1 + Int(compute_odd)
    total = nq * (eigenpair_passes)

    # define one local advance
    p = show_progress && total > 0 ?
        Progress(total; desc="Computing spectral basis") :
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
                                 qdata.idx64, qdata.qs64, domain, Ns, β;
                                 Nmax=Nmax,
                                 parallel=parallel,
                                 advance_progress=advance_progress,
                                 compute_odd=compute_odd,
                                 sweep_chunk_size=sweep_chunk_size)

        _scatter_spectral_group!(qs, an, bn, Ak, Bk,
                                 qdata.idxbig, qdata.qsbig, domain, Ns, β;
                                 Nmax=Nmax,
                                 parallel=parallel,
                                 advance_progress=advance_progress,
                                 compute_odd=compute_odd,
                                 sweep_chunk_size=sweep_chunk_size)

        _scatter_spectral_group!(qs, an, bn, Ak, Bk,
                                 qdata.idxHR, qdata.qsHR, domain, Ns, β;
                                 Nmax=Nmax,
                                 advance_progress=advance_progress,
                                 parallel=parallel,
                                 compute_odd=compute_odd,
                                 sweep_chunk_size=sweep_chunk_size)
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
Collect eigenpairs for the alongstream wavenumber sweep.


"""
function spectral_basis(U::AbstractVector, Pe, domain::Domain, Ns;
                        precision_policy=:float64, qc1::Union{Nothing,Real}=nothing, 
                        qc2::Union{Nothing,Real}=nothing,
                        compute_odd::Bool=false,
                        parallel::Bool = true,
                        show_progress::Bool=true, 
                        sweep_chunk_size::Union{Nothing,Integer}=nothing)
    _, β0, β = flow_decomposition(U, domain)
    data = _collect_spectral_data(domain, Pe, Ns, β;
                                  precision_policy=precision_policy, qc1=qc1, qc2=qc2,
                                  parallel=parallel,
                                  show_progress=show_progress,
                                  compute_odd=compute_odd,
                                  sweep_chunk_size=sweep_chunk_size)

    return SpectralBasis(data.qs, β0, β, data.an, data.bn, data.Ak, data.Bk)
end


"""
Collect eigenpairs for a predefined flow over the alongstream wavenumber sweep.

"""
function spectral_basis(flow::Symbol, Pe, domain::Domain, Ns::Union{Nothing,Integer}=nothing;
                        precision_policy=:float64, qc1::Union{Nothing,Real}=nothing,
                        qc2::Union{Nothing,Real}=nothing,
                        compute_odd::Bool=false,
                        parallel::Bool=true,
                        show_progress::Bool=true, 
                        sweep_chunk_size::Union{Nothing,Integer}=nothing)
    _, β0, β = flow_decomposition(flow, domain)
    data = _collect_spectral_data(domain, Pe, Ns, β;
                                  precision_policy=precision_policy, qc1=qc1, qc2=qc2, 
                                  parallel=parallel,
                                  show_progress=show_progress,
                                  compute_odd=compute_odd,
                                  sweep_chunk_size=sweep_chunk_size)

    return SpectralBasis(data.qs, β0, β, data.an, data.bn, data.Ak, data.Bk)
end
