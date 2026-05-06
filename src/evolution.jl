
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


function _require_basis_eigenfunctions(basis)
    basis.Φe === nothing &&
        throw(ArgumentError("basis must include eigenfunctions; call spectral_basis(...; compute_eigenfunctions=true)"))
    return nothing
end


function _prepare_mode_work(dom, basis, γ, γ_o=nothing)
    _require_basis_eigenfunctions(basis)

    nk = length(dom.K)

    etails  = Vector{Any}(undef, nk)
    work   = Vector{Any}(undef, nk)

    if γ_o !== nothing
        otails  = Vector{Any}(undef, nk)    
    else
        e_o = nothing
        C_o = nothing
        Umat_o = nothing
        f_o = nothing
        otails  = nothing   
    end

    for k in 1:nk
        Φe = basis.Φe[k]
        Ak = basis.Ak[k]
        an = basis.an[k]

        # this may be empty
        Φo = basis.Φo[k]
        Bk = basis.Bk[k]
        bn = basis.bn[k]

        ny, Nn = size(Φe) # same size as odd

        etails[k] = cosine_tail_data(dom.y, γ, Nn)

        e = Vector{eltype(an)}(undef, Nn)
        C = similar(Ak)
        Umat = Matrix{promote_type(eltype(Φe), eltype(Ak), eltype(an))}(undef, ny, Nn)
        Tf = promote_type(eltype(Umat), eltype(γ))
        f = Vector{Tf}(undef, ny)

        if γ_o !== nothing
            otails[k] = sine_tail_data(dom.y, γ_o, Nn)
            e_o = Vector{eltype(bn)}(undef, Nn)
            C_o = similar(Bk)
            Umat_o = Matrix{promote_type(eltype(Φo), eltype(Bk), eltype(bn))}(undef, ny, Nn)
            f_o = Vector{Tf}(undef, ny)
        end

        work[k] = (e=e, C=C, U=Umat, f=f, e_o=e_o, C_o=C_o, U_o=Umat_o, f_o=f_o)
    end

    return etails, work, otails
end


function _validate_reconstruction(reconstruction::Symbol)
    reconstruction === :eigenfunctions && return nothing
    reconstruction === :coefficients && return nothing
    throw(ArgumentError("reconstruction must be :eigenfunctions or :coefficients; got $(repr(reconstruction))"))
end


function _cosine_y_basis(y::AbstractVector, R::Integer)
    B = Matrix{Float64}(undef, length(y), R)
    for r in 0:(R-1)
        @inbounds for iy in eachindex(y)
            B[iy, r + 1] = cos(2r * y[iy])
        end
    end
    return B
end

function _sine_y_basis(y::AbstractVector, R::Integer)
    B = Matrix{Float64}(undef, length(y), R)
    for r in 0:(R-1)
        @inbounds for iy in eachindex(y)
            B[iy, r + 1] = sin(2(r+1) * y[iy])
        end
    end
    return B
end

function _prepare_coefficient_mode_work_point(dom, basis, γ, Pe, conj_spectrum, k)
    Ak = conj_spectrum ? conj.(basis.Ak[k]) : basis.Ak[k]

    R, Nn = size(Ak)
    Kk = dom.K[k]

    λ = ComplexF64.(basis.an[k])
    if conj_spectrum
        λ .= conj.(λ)
    end
    βshift = im * (Float64(basis.β0) / 2) * Kk * Float64(Pe)
    @. λ = λ / 4 + Kk^2 + βshift

    Nuse = min(R, length(γ))
    γlow = @view γ[1:Nuse]

    Tproj = promote_type(eltype(Ak), eltype(γ))
    projection = Vector{Tproj}(undef, Nn)
    if Nuse > 0
        mul!(projection, transpose(view(Ak, 1:Nuse, :)), γlow)
    else
        fill!(projection, zero(Tproj))
    end

    By = _cosine_y_basis(dom.y, R)
    Tfield = promote_type(eltype(By), eltype(projection))

    e = Vector{ComplexF64}(undef, Nn)
    modal = similar(projection)
    coeff_y = Vector{Tproj}(undef, R)
    f = Vector{Tfield}(undef, length(dom.y))

    tail = cosine_tail_data(dom.y, γ, Nn)

    return (Ak=Ak, λ=λ, projection=projection, By=By, e=e,
            modal=modal, coeff_y=coeff_y, f=f, tail=tail)
end


function _prepare_coefficient_mode_work(dom, basis, γ, Pe;
                                        conj_spectrum::Bool=false)
    nk = length(dom.K)
    work = Vector{Any}(undef, nk)

    for k in 1:nk
        work[k] = _prepare_coefficient_mode_work_point(dom, basis, γ, Pe,
                                                       conj_spectrum, k)
    end

    return work
end


function _time_chunks(nt::Integer, nchunks::Integer)
    chunks = Vector{UnitRange{Int}}(undef, nchunks)
    for i in 1:nchunks
        first = ((i - 1) * nt) ÷ nchunks + 1
        last = (i * nt) ÷ nchunks
        chunks[i] = first:last
    end
    return chunks
end


function _fetch_all_tasks(tasks)
    task_error = nothing
    task_backtrace = nothing

    for task in tasks
        try
            fetch(task)
        catch err
            if task_error === nothing
                task_error = err
                task_backtrace = catch_backtrace()
            end
        end
    end

    task_error === nothing || Base.throw_with_backtrace(task_error, task_backtrace)
    return nothing
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


function _solve_explicit_time!(T, it, t, spectrum, tails, work, otails,
                               dom, basis, γ, γ_o, Ccoeffs, Pe, conj_spectrum,
                               x0, nx, ny, nk)
    fill!(spectrum, 0)

    for k in 1:nk
        Φe = basis.Φe[k]
        Ak = basis.Ak[k]
        an = basis.an[k]

        Φo = basis.Φo[k]
        Bk = basis.Bk[k]
        bn = basis.bn[k]

        Kk = dom.K[k]

        e = work[k].e
        C = work[k].C
        Umat = work[k].U
        f = work[k].f

        # here work odd pairs
        e_o = work[k].e_o
        C_o = work[k].C_o
        Umat_o = work[k].U_o
        f_o = work[k].f_o

        m_tail, Atail, γtail, tmp = tails[k]

        m_otail, Btail, γotail, otmp = otails[k]

        e .= exp.(-t .* ((an ./ 4) .+ Kk^2 .+ (im * (basis.β0/2) * Kk * Pe)))
        C .= Ak .* reshape(e, 1, :)
        mul!(Umat, Φe, transpose(C))

        e_o .= exp.(-t .* ((bn ./ 4) .+ Kk^2 .+ (im * (basis.β0/2) * Kk * Pe)))
        C_o .= Bk .* reshape(e_o, 1, :)
        mul!(Umat_o, Φo, transpose(C_o))

        reconstruct!(f, Umat, γ, Atail, γtail, m_tail, tmp, t, Kk)

        if γ_o !== nothing
            reconstruct!(f_o, Umat_o, γ_o, Btail, γotail, m_otail, otmp, t, Kk)
        end

        # Modal reconstruction may use high precision; FFTW x-inversion is Float64.
        phase0 = Ccoeffs[k] * cis(Kk * x0)
        col = @view spectrum[:, k]
        if _irfft_self_conjugate_mode(k, nx)
            @inbounds @simd for iy in 1:ny
                col[iy] = nx * real(ComplexF64((f[iy] + f_o[iy]) * phase0))
            end
        else
            scale = nx / 2
            @inbounds @simd for iy in 1:ny
                col[iy] = scale * ComplexF64((f[iy] + f_o[iy]) * phase0)
            end
        end
    end

    @views T[it, :, :] .= irfft(spectrum, nx, 2)
    return nothing
end


function _add_tail!(f, tail, t, Kk)
    m_tail, Atail, γtail, tmp = tail
    isempty(m_tail) && return nothing

    k2 = Kk^2
    @inbounds for j in eachindex(m_tail)
        tmp[j] = γtail[j] * exp(-Float64(t) * (Float64(m_tail[j])^2 + k2))
    end
    mul!(f, Atail, tmp, one(eltype(f)), one(eltype(f)))

    return nothing
end


function _solve_explicit_time_coefficients!(T, it, t, spectrum, work,
                                            dom, Ccoeffs, x0, nx, ny, nk)
    fill!(spectrum, 0)
    t64 = Float64(t)

    for k in 1:nk
        wk = work[k]
        Kk = dom.K[k]

        @. wk.e = exp(-t64 * wk.λ)
        @. wk.modal = wk.projection * wk.e
        mul!(wk.coeff_y, wk.Ak, wk.modal)
        mul!(wk.f, wk.By, wk.coeff_y)
        _add_tail!(wk.f, wk.tail, t64, Kk)

        phase0 = Ccoeffs[k] * cis(Kk * x0)
        col = @view spectrum[:, k]
        if _irfft_self_conjugate_mode(k, nx)
            @inbounds @simd for iy in 1:ny
                col[iy] = nx * real(ComplexF64(wk.f[iy] * phase0))
            end
        else
            scale = nx / 2
            @inbounds @simd for iy in 1:ny
                col[iy] = scale * ComplexF64(wk.f[iy] * phase0)
            end
        end
    end

    @views T[it, :, :] .= irfft(spectrum, nx, 2)
    return nothing
end


function solve_explicit(dom, basis, γ, γ_o, Ccoeffs, tvals, Pe;
                        show_progress::Bool=true,
                        parallel::Bool=true,
                        reconstruction::Symbol=:coefficients)
    _validate_reconstruction(reconstruction)

    conj_spectrum = Pe < 0
    nt = length(tvals)
    ny = length(dom.y)
    nx = length(dom.x)
    nk = length(dom.K)

    T = zeros(Float64, nt, ny, nx)

    x0 = dom.x[1]
    reconstruction === :eigenfunctions && _require_basis_eigenfunctions(basis)

    # define progressmeter instance
    p = show_progress && nt > 0 ?
        Progress(nt; desc="Computing time-evolving solution") :
        nothing
    
    processed = Ref(0)

    advance_progress = p === nothing ? nothing : () -> begin
        processed[] += 1
        update!(p, processed[])
        return nothing
    end

    completed=false

    try
        if parallel && Threads.nthreads() > 1 && nt > 1
            tbuf = collect(tvals)
            nworkers = min(Threads.nthreads(), nt)
            chunks = _time_chunks(nt, nworkers)
            progress_lock = ReentrantLock()

            if advance_progress !== nothing
                advance_progress = () -> begin
                    lock(progress_lock)
                    try
                        processed[] += 1
                        update!(p, processed[])
                    finally
                        unlock(progress_lock)
                    end
                    return nothing
                end
            end

            tasks = map(chunks) do chunk
                Threads.@spawn begin
                    if reconstruction === :coefficients
                        local_work = _prepare_coefficient_mode_work(dom, basis, γ, Pe;
                                                                    conj_spectrum=conj_spectrum)
                    else
                        local_tails, local_work, local_otails = _prepare_mode_work(dom, basis, γ, γ_o)
                    end
                    local_spectrum = zeros(ComplexF64, ny, nx ÷ 2 + 1)

                    for it in chunk
                        if reconstruction === :coefficients
                            _solve_explicit_time_coefficients!(T, it, tbuf[it],
                                                               local_spectrum,
                                                               local_work, dom,
                                                               Ccoeffs, x0,
                                                               nx, ny, nk)
                        else
                            _solve_explicit_time!(T, it, tbuf[it], local_spectrum,
                                                  local_tails, local_work, local_otails, dom, basis,
                                                  γ, γ_o, Ccoeffs, Pe, conj_spectrum,
                                                  x0, nx, ny, nk)
                        end
                        advance_progress === nothing || advance_progress()
                    end
                end
            end

            _fetch_all_tasks(tasks)
        else
            if reconstruction === :coefficients
                work = _prepare_coefficient_mode_work(dom, basis, γ, Pe;
                                                      conj_spectrum=conj_spectrum)
            else
                tails, work, otails = _prepare_mode_work(dom, basis, γ, γ_o)
            end
            spectrum = zeros(ComplexF64, ny, nx ÷ 2 + 1)

            for (it, t) in enumerate(tvals)
                if reconstruction === :coefficients
                    _solve_explicit_time_coefficients!(T, it, t, spectrum,
                                                       work, dom, Ccoeffs, x0,
                                                       nx, ny, nk)
                else
                    _solve_explicit_time!(T, it, t, spectrum, tails, work, otails, dom, basis,
                                          γ, γ_o, Ccoeffs, Pe, conj_spectrum, x0, nx, ny, nk)
                end
                advance_progress === nothing || advance_progress()
            end
        end
        completed = true
    finally
        if p !== nothing
            if completed
                finish!(p)
            else
                cancel(p, "Time evolution aborted")
            end
        end
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


function prepare_explicit_avg(dom, U::AbstractVector, γ, Pe, Ns;
                              precision_policy=:float64, qc1::Union{Nothing,Real}=nothing,
                              qc2::Union{Nothing,Real}=nothing,
                              parallel::Bool=true,
                              show_progress::Bool=true, 
                              sweep_chunk_size::Union{Nothing,Integer}=nothing)
    conj_spectrum = Pe < 0
    Peabs = abs(Pe)
    Nmax = length(γ)

    _, β0, β = flow_decomposition(U, dom)
    data = _collect_spectral_data(dom, Peabs, Ns, β;
                                  Nmax=Nmax,
                                  precision_policy=precision_policy,
                                  qc1=qc1, qc2=qc2,
                                  parallel=parallel,
                                  show_progress=show_progress,
                                  compute_eigenfunctions=false,
                                  compute_odd=false, 
                                  sweep_chunk_size=sweep_chunk_size) # this is not true always

    work = prepare_mode_data_avg(dom, data.Ak, data.an, γ, β0, Pe;
                                 conj_spectrum=conj_spectrum)

    return (; β0, work)
end


function prepare_explicit_avg(dom, flow::Symbol, γ, Pe, Ns;
                              precision_policy=:float64, qc1::Union{Nothing,Real}=nothing,
                              qc2::Union{Nothing,Real}=nothing,
                              parallel::Bool=true,
                              show_progress::Bool=true, 
                              sweep_chunk_size::Union{Nothing,Integer}=nothing)
    conj_spectrum = Pe < 0
    Peabs = abs(Pe)
    Nmax = length(γ)

    _, β0, β = flow_decomposition(flow, dom)
    data = _collect_spectral_data(dom, Peabs, Ns, β;
                                  Nmax=Nmax,
                                  precision_policy=precision_policy,
                                  qc1=qc1, qc2=qc2,
                                  parallel=parallel,
                                  show_progress=show_progress,
                                  compute_eigenfunctions=false,
                                  compute_odd=false, 
                                  sweep_chunk_size=sweep_chunk_size)

    work = prepare_mode_data_avg(dom, data.Ak, data.an, γ, β0, Pe;
                                 conj_spectrum=conj_spectrum)

    return (; β0, work)
end



function _avg_work_scratch(work)
    return [AvgModeWork(w.A0, w.λ, w.γproj, similar(w.e)) for w in work]
end


function _evolve_explicit_avg_time!(T, it, t, spec, work, dom, Ccoeffs, x0, nx)
    fill!(spec, 0)

    for k in eachindex(dom.K)
        wk = work[k]

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

    @views T[it, :] .= irfft(spec, nx)
    return nothing
end


function evolve_explicit_avg(dom, prep, Ccoeffs, tvals;
                             show_progress::Bool=true,
                             parallel::Bool=true)
    nt = length(tvals)
    nx = length(dom.x)

    T = zeros(Float64, nt, nx)
    spec = zeros(ComplexF64, nx ÷ 2 + 1)

    x0 = dom.x[1]

    # define progressmeter instance
    p = show_progress && nt > 0 ?
        Progress(nt; desc="Computing time-evolving solution") :
        nothing
    
    processed = Ref(0)

    advance_progress = p === nothing ? nothing : () -> begin
        processed[] += 1
        update!(p, processed[])
        return nothing
    end

    completed=false

    try
        if parallel && Threads.nthreads() > 1 && nt > 1
            tbuf = collect(tvals)
            nworkers = min(Threads.nthreads(), nt)
            chunks = _time_chunks(nt, nworkers)
            progress_lock = ReentrantLock()

            if advance_progress !== nothing
                advance_progress = () -> begin
                    lock(progress_lock)
                    try
                        processed[] += 1
                        update!(p, processed[])
                    finally
                        unlock(progress_lock)
                    end
                    return nothing
                end
            end

            tasks = map(chunks) do chunk
                Threads.@spawn begin
                    local_work = _avg_work_scratch(prep.work)
                    local_spec = zeros(ComplexF64, nx ÷ 2 + 1)

                    for it in chunk
                        _evolve_explicit_avg_time!(T, it, tbuf[it], local_spec,
                                                   local_work, dom, Ccoeffs, x0, nx)
                        advance_progress === nothing || advance_progress()
                    end
                end
            end

            _fetch_all_tasks(tasks)
        else
            for (it, t) in enumerate(tvals)
                _evolve_explicit_avg_time!(T, it, t, spec, prep.work, dom,
                                           Ccoeffs, x0, nx)
                advance_progress === nothing || advance_progress()
            end
        end
        completed = true
    finally
        if p !== nothing
            if completed
                finish!(p)
            else
                cancel(p, "Time evolution aborted")
            end
        end
    end

    return T
end
