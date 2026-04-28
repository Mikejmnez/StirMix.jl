
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


function prepare_mode_data(dom, basis, γ, Ccoeffs)
    nk = length(dom.K)

    phases = Vector{Vector{ComplexF64}}(undef, nk)
    tails  = Vector{Any}(undef, nk)
    work   = Vector{Any}(undef, nk)

    for k in 1:nk
        Φe = basis.Φe[k]
        Ak = basis.Ak[k]
        an = basis.an[k]

        ny, Nn = size(Φe)

        phases[k] = Ccoeffs[k] .* exp.(im .* dom.K[k] .* dom.x)
        tails[k]  = cosine_tail_data(dom.y, γ, Nn)

        e = Vector{eltype(an)}(undef, Nn)
        C = similar(Ak)
        Umat = Matrix{promote_type(eltype(Φe), eltype(Ak), eltype(an))}(undef, ny, Nn)
        Tf = promote_type(eltype(Umat), eltype(γ))
        f = Vector{Tf}(undef, ny)

        work[k] = (e=e, C=C, U=Umat, f=f)
    end

    return phases, tails, work
end


function prepare_mode_data_avg(dom, Ak, avals, γ, β0, Pe; conj_spectrum=false)
    nk = length(dom.K)

    Tmode = promote_type(eltype(Ak[1]), eltype(avals[1]), eltype(float(γ[1])))
    Tmode = promote_type(eltype(Ak[1]), eltype(avals[1]), eltype(γ))

    work = Vector{AvgModeWork{Tmode}}(undef, nk)

    βshift = im * (β0 / 2) * Pe

    for k in 1:nk
        Akk = conj_spectrum ? conj.(Ak[k])    : Ak[k]
        ank = conj_spectrum ? conj.(avals[k]) : avals[k]

        Nn = length(ank)
        A0 = Vector{Tmode}(undef, Nn)
        γproj = Vector{Tmode}(undef, Nn)
        e = Vector{Tmode}(undef, Nn)
        λ = Vector{Tmode}(undef, Nn)

        Nγ = min(length(γ), size(Akk, 1))
        Kk = dom.K[k]

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

    phases, tails, work = prepare_mode_data(dom, basis, γ, Ccoeffs)

    Tt = zeros(Float64, ny, nx)

    for (it, t) in enumerate(tvals)
        fill!(Tt, 0.0)

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

            BLAS.ger!(1.0, real(f), real(phases[k]), Tt)
            BLAS.ger!(-1.0, imag(f), imag(phases[k]), Tt)
        end

        T[it, :, :] .= Tt;
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

    qs, _, β0, β = flow_decomposition(U, Peabs, dom)
    _, avals, Ak = collect_sweep_eigen(Even, qs, Ns, β; Nmax=Nmax)

    work = prepare_mode_data_avg(dom, Ak, avals, γ, β0, Pe;
                                 conj_spectrum=conj_spectrum)

    return (; β0, work)
end


function prepare_explicit_avg(dom, flow::Symbol, γ, Pe, Ns)
    conj_spectrum = Pe < 0
    Peabs = abs(Pe)
    Nmax = length(γ)

    qs, _, β0, β = flow_decomposition(flow, Peabs, dom)
    _, avals, Ak = collect_sweep_eigen(Even, qs, Ns, β; Nmax=Nmax)

    work = prepare_mode_data_avg(dom, Ak, avals, γ, β0, Pe;
                                 conj_spectrum=conj_spectrum)

    return (; β0, work)
end


function prepare_phase_data_avg(dom, Ccoeffs)
    nk = length(dom.K)
    x = dom.x
    x1 = x[1]
    dx = x[2] - x[1]

    ωr  = Vector{Float64}(undef, nk)
    ωi  = Vector{Float64}(undef, nk)
    p0r = Vector{Float64}(undef, nk)
    p0i = Vector{Float64}(undef, nk)

    for k in 1:nk
        Kk = dom.K[k]

        ωr[k] = cos(Kk * dx)
        ωi[k] = sin(Kk * dx)

        p0 = Ccoeffs[k] * cis(Kk * x1)
        p0r[k] = real(p0)
        p0i[k] = imag(p0)
    end

    return (; ωr, ωi, p0r, p0i)
end


using Base.Threads

function evolve_explicit_avg(dom, prep, Ccoeffs, tvals)
    nt = length(tvals)
    nx = length(dom.x)
    nk = length(dom.K)
    nthreads_ = nthreads()

    T = zeros(Float64, nt, nx)

    # One private Tt buffer per thread
    Tlocal = [zeros(Float64, nx) for _ in 1:nthreads_]

    work  = prep.work
    phase = prepare_phase_data_avg(dom, Ccoeffs)

    for (it, t) in enumerate(tvals)
        # reset thread-local accumulators
        for buf in Tlocal
            fill!(buf, 0.0)
        end

        @threads for k in 1:nk
            tid = threadid()
            Tt = Tlocal[tid]

            wk = work[k]
            A0    = wk.A0
            λ     = wk.λ
            γproj = wk.γproj
            e     = wk.e

            @. e = exp(-t * λ)

            favg = zero(eltype(e))
            @inbounds @simd for n in eachindex(e)
                favg += A0[n] * e[n] * γproj[n]
            end

            ar = real(favg)
            ai = imag(favg)

            pr = phase.p0r[k]
            pi = phase.p0i[k]
            wr = phase.ωr[k]
            wi = phase.ωi[k]

            @inbounds for ix in 1:nx
                Tt[ix] += ar * pr - ai * pi
                pr, pi = pr * wr - pi * wi, pr * wi + pi * wr
            end
        end

        # reduce thread-local buffers into output row
        row = @view T[it, :]
        fill!(row, 0.0)
        for tid in 1:nthreads_
            @inbounds @simd for ix in 1:nx
                row[ix] += Tlocal[tid][ix]
            end
        end
    end

    return T
end
