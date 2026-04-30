
"""
    precompute_x_phase(K, x, C, β0, Pe)

Precompute
    P[k, ix] = C[k] * exp(im*K[k]*x[ix]) * exp(im*β0*K[k]*Pe)

Returns a matrix P of size (nk, nx).
"""
function precompute_x_phase(K::AbstractVector, x::AbstractVector,
                            C::AbstractVector, β0, Pe)
    nk = length(K)
    nx = length(x)
    @assert length(C) == nk

    P = Matrix{ComplexF64}(undef, nk, nx)
    for k in 1:nk
        amp = C[k] * exp(im * β0 * K[k] * Pe)
        @inbounds P[k, :] .= amp .* exp.(im .* K[k] .* x)
    end
    return P
end


"""
    reconstruct_x!(T, Tc, F, P)

Given
- F  : (ny, nk) modal y-solutions
- P  : (nk, nx) precomputed x-phase matrix

compute
    T = real(F * P)

with Tc as complex workspace.
"""
function reconstruct_x!(T::AbstractMatrix{<:Real},
                        Tc::AbstractMatrix{ComplexF64},
                        F::AbstractMatrix{ComplexF64},
                        P::AbstractMatrix{ComplexF64})
    mul!(Tc, F, P)
    @. T = real(Tc)
    return T
end


@inline function _irfft_self_conjugate_mode(k::Integer, nx::Integer)
    m = k - 1
    return m == 0 || (iseven(nx) && m == nx ÷ 2)
end


function _prepare_mode_work(dom, basis, γ)
    basis.Φe === nothing &&
        throw(ArgumentError("basis must include eigenfunctions; call spectral_basis(...; compute_eigenfunctions=true)"))

    nk = length(dom.K)

    tails  = Vector{Any}(undef, nk)
    work   = Vector{Any}(undef, nk)

    for k in 1:nk
        Φe = basis.Φe[k]
        Ak = basis.Ak[k]
        an = basis.an[k]

        ny, Nn = size(Φe)

        tails[k]  = cosine_tail_data(dom.y, γ, Nn)

        e = Vector{eltype(an)}(undef, Nn)
        C = similar(Ak)
        Umat = Matrix{promote_type(eltype(Φe), eltype(Ak), eltype(an))}(undef, ny, Nn)
        Tf = promote_type(eltype(Umat), eltype(γ))
        f = Vector{Tf}(undef, ny)

        work[k] = (e=e, C=C, U=Umat, f=f)
    end

    return tails, work
end


function prepare_mode_data(dom, basis, γ, Ccoeffs)
    nk = length(dom.K)

    phases = Vector{Vector{ComplexF64}}(undef, nk)
    for k in 1:nk
        phases[k] = Ccoeffs[k] .* exp.(im .* dom.K[k] .* dom.x)
    end

    tails, work = _prepare_mode_work(dom, basis, γ)

    return phases, tails, work
end


function prepare_mode_data_avg(dom, Ak, avals, γ, β0, Pe; conj_spectrum=false)
    nk = length(dom.K)

    work = Vector{AvgModeWork}(undef, nk)

    βshift = im * (β0 / 2) * Pe

    for k in 1:nk
        Akk = conj_spectrum ? conj.(Ak[k])    : Ak[k]
        ank = conj_spectrum ? conj.(avals[k]) : avals[k]

        Nn = length(ank)
        Kk = dom.K[k]
        Tmode = promote_type(eltype(Akk), eltype(ank), eltype(γ),
                             typeof(βshift * Kk), typeof(Kk^2))

        A0 = Vector{Tmode}(undef, Nn)
        γproj = Vector{Tmode}(undef, Nn)
        e = Vector{Tmode}(undef, Nn)
        λ = Vector{Tmode}(undef, Nn)

        Nγ = min(length(γ), size(Akk, 1))

        copyto!(A0, vec(Akk[1, :]))

        mul!(γproj, transpose(view(Akk, 1:Nγ, :)), view(γ, 1:Nγ))
        
        @. λ = ank / 4 + Kk^2 + βshift * Kk

        work[k] = AvgModeWork(A0, λ, γproj, e)
    end

    return work
end


function solve_explicit(dom, basis, γ, Ccoeffs, tvals, Pe)
    conj_spectrum = Pe < 0
    nt = length(tvals)
    ny = length(dom.y)
    nx = length(dom.x)
    nk = length(dom.K)

    T = zeros(Float64, nt, ny, nx)

    tails, work = _prepare_mode_work(dom, basis, γ)

    spectrum = zeros(ComplexF64, ny, nx ÷ 2 + 1)
    x0 = dom.x[1]

    for (it, t) in enumerate(tvals)
        fill!(spectrum, 0)

        for k in 1:nk
            Φe = conj_spectrum ? conj.(basis.Φe[k]) : basis.Φe[k]
            Ak = conj_spectrum ? conj.(basis.Ak[k]) : basis.Ak[k]
            an = conj_spectrum ? conj.(basis.an[k]) : basis.an[k]
            Kk = dom.K[k]

            e = work[k].e
            C = work[k].C
            Umat = work[k].U
            f = work[k].f

            m_tail, Atail, γtail, tmp = tails[k]

            e .= exp.(-t .* ((an ./ 4) .+ Kk^2 .+ (im * (basis.β0/2) * Kk * Pe)))
            C .= Ak .* reshape(e, 1, :)
            mul!(Umat, Φe, transpose(C))

            reconstruct!(f, Umat, γ, Atail, γtail, m_tail, tmp, t, Kk)

            # Modal reconstruction may use high precision; FFTW x-inversion is Float64.
            phase0 = Ccoeffs[k] * cis(Kk * x0)
            col = @view spectrum[:, k]
            if _irfft_self_conjugate_mode(k, nx)
                @inbounds @simd for iy in 1:ny
                    col[iy] = nx * real(ComplexF64(f[iy] * phase0))
                end
            else
                scale = nx / 2
                @inbounds @simd for iy in 1:ny
                    col[iy] = scale * ComplexF64(f[iy] * phase0)
                end
            end
        end

        @views T[it, :, :] .= irfft(spectrum, nx, 2)
    end

    return T
end


"""
    time_resample(ft::AbstractVector{<:Real}; nt::Integer=0)

Discretize a real-valued periodic signal `ft` into a piecewise-constant signal
with levels spaced by

    Δ = 1 / 2^nt

over the interval [-1, 1].

Returns `(nft, vals, ivals)` where

- `nft`   : discretized signal
- `vals`  : all allowed discrete values, ordered increasingly
- `ivals` : indices such that `vals[ivals[i]] == nft[i]`

Notes:
- `ivals` uses Julia's 1-based indexing.
- This version uses nearest-level rounding, which reproduces the Python
  midpoint logic efficiently.
"""
function time_resample(ft::AbstractVector{<:Real}; nt::Integer=0)
    nt < 0 && throw(ArgumentError("nt must be >= 0"))

    T = float(eltype(ft))
    Δ = one(T) / T(2^nt)

    # Allowed values: -1, ..., -Δ, 0, Δ, ..., 1
    vals = collect(-one(T):Δ:one(T))

    nft = Vector{T}(undef, length(ft))
    ivals = Vector{Int}(undef, length(ft))

    offset = Int(round(one(T) / Δ)) + 1   # maps 0 -> center index

    @inbounds for i in eachindex(ft)
        # nearest allowed level
        y = clamp(Δ * round(T(ft[i]) / Δ), -one(T), one(T))
        nft[i] = y

        # map y to its position in vals
        ivals[i] = Int(round(y / Δ)) + offset
    end

    return nft, vals, ivals
end


function prepare_explicit_avg(dom, U::AbstractVector, γ, Pe, Ns)
    conj_spectrum = Pe < 0
    Peabs = abs(Pe)
    Nmax = length(γ)

    _, β0, β = flow_decomposition(U, Peabs, dom)
    data = _collect_spectral_data(dom, Peabs, Ns, β;
                                  Nmax=Nmax,
                                  compute_eigenfunctions=false,
                                  compute_odd=false)

    work = prepare_mode_data_avg(dom, data.Ak, data.an, γ, β0, Pe;
                                 conj_spectrum=conj_spectrum)

    return (; β0, work)
end


function prepare_explicit_avg(dom, flow::Symbol, γ, Pe, Ns)
    conj_spectrum = Pe < 0
    Peabs = abs(Pe)
    Nmax = length(γ)

    _, β0, β = flow_decomposition(flow, Peabs, dom)
    data = _collect_spectral_data(dom, Peabs, Ns, β;
                                  Nmax=Nmax,
                                  compute_eigenfunctions=false,
                                  compute_odd=false)

    work = prepare_mode_data_avg(dom, data.Ak, data.an, γ, β0, Pe;
                                 conj_spectrum=conj_spectrum)

    return (; β0, work)
end



function evolve_explicit_avg(dom, prep, Ccoeffs, tvals)
    nt = length(tvals)
    nx = length(dom.x)

    T = zeros(Float64, nt, nx)
    spec = zeros(ComplexF64, nx ÷ 2 + 1)

    x0 = dom.x[1]

    for (it, t) in enumerate(tvals)
        fill!(spec, 0)

        for k in eachindex(dom.K)
            wk = prep.work[k]

            @. wk.e = exp(-t * wk.λ)

            # Modal averages may use high precision; 
            # FFTW reconstruction is intentionally Float64.
            favg = zero(eltype(wk.e))
            @inbounds @simd for n in eachindex(wk.e)
                favg += wk.A0[n] * wk.e[n] * wk.γproj[n]
            end

            # Big/complex arithmetic may happen above; FFTW boundary is here.
            z = ComplexF64(favg * Ccoeffs[k] * cis(dom.K[k] * x0))

            if _irfft_self_conjugate_mode(k, nx)
                spec[k] = nx * real(z)
            else
                spec[k] = (nx / 2) * z
            end
        end

        T[it, :] .= irfft(spec, nx)
    end

    return T
end
