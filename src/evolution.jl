@inline function _irfft_self_conjugate_mode(k::Integer, nx::Integer)
    m = k - 1
    return m == 0 || (iseven(nx) && m == nx ÷ 2)
end


function _require_basis_odd_coefficients(basis)
    (basis.Bk === nothing || basis.bn === nothing) &&
        throw(ArgumentError("basis must include odd coefficient data; call spectral_basis(...; compute_odd=true)"))

    return nothing
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

function _prepare_coefficient_mode_work_point(dom, basis, γ, γ_o, Pe, k)
    Ak = basis.Ak[k]

    R, Nn = size(Ak)
    Kk = 2π * dom.K[k]
    βshift = im * (Float64(basis.β0) / 2) * Kk * Float64(Pe)

    λ = ComplexF64.(basis.an[k])
    @. λ =  λ / 4 + Kk^2 + βshift

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

    Bk = nothing
    λ_o = nothing
    projection_o = nothing
    By_o = nothing
    e_o = nothing
    modal_o = nothing
    coeff_y_o = nothing
    f_o = nothing
    tail_o = nothing

    if any(γ_o .> 1e-6)
        Bk = basis.Bk[k]
        R_o, Nno = size(Bk)

        λ_o = ComplexF64.(basis.bn[k])
        @. λ_o = λ_o / 4 + Kk^2 + βshift

        Nuse_o = min(R_o, length(γ_o))
        γlow_o = @view γ_o[1:Nuse_o]

        Tproj_o = promote_type(eltype(Bk), eltype(γ_o))
        projection_o = Vector{Tproj_o}(undef, Nno)
        if Nuse_o > 0
            mul!(projection_o, transpose(view(Bk, 1:Nuse_o, :)), γlow_o)
        else
            fill!(projection_o, zero(Tproj_o))
        end

        By_o = _sine_y_basis(dom.y, R_o)
        Tfield_o = promote_type(eltype(By_o), eltype(projection_o))

        e_o = Vector{ComplexF64}(undef, Nno)
        modal_o = similar(projection_o)
        coeff_y_o = Vector{Tproj_o}(undef, R_o)
        f_o = Vector{Tfield_o}(undef, length(dom.y))

        tail_o = sine_tail_data(dom.y, γ_o, R_o)
    end

    return (Ak=Ak, λ=λ, projection=projection, By=By, e=e,
            modal=modal, coeff_y=coeff_y, f=f, tail=tail,
            Bk=Bk, λ_o=λ_o, projection_o=projection_o, By_o=By_o,
            e_o=e_o, modal_o=modal_o, coeff_y_o=coeff_y_o, f_o=f_o,
            tail_o=tail_o)
end


function _prepare_coefficient_mode_work(dom, basis, γ, γ_o, Pe)
    any(γ_o .> 1e-6) && _require_basis_odd_coefficients(basis)

    nk = length(dom.K)
    work = Vector{Any}(undef, nk)

    for k in 1:nk
        work[k] = _prepare_coefficient_mode_work_point(dom, basis, γ, γ_o, Pe, k)
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


function prepare_mode_data_avg(dom, Ak, avals, γ, β0, Pe)
    nk = length(dom.K)

    work = Vector{AvgModeWork}(undef, nk)

    βshift = im * (β0 / 2) * Pe

    for k in 1:nk
        Akk = Ak[k]
        ank = avals[k]

        Nn = length(ank)
        Kk = 2π * dom.K[k]
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


function _add_tail!(f, tail, t, Kk, fac)
    m_tail, Atail, γtail, tmp = tail
    isempty(m_tail) && return nothing

    k2 = Kk^2
    @inbounds for j in eachindex(m_tail)
        tmp[j] = γtail[j] * exp(-Float64(t) * (fac * Float64(m_tail[j])^2 + k2))
    end
    mul!(f, Atail, tmp, one(eltype(f)), one(eltype(f)))

    return nothing
end


function _solve_explicit_time!(T, it, t, spectrum, work,
                                            dom, Ccoeffs, x0, nx, ny, nk, fac)
    fill!(spectrum, 0)
    t64 = Float64(t)
    _ny = size(T,2)

    for k in 1:nk
        wk = work[k]
        Kk = 2π * dom.K[k]

        @. wk.e = exp(-t64 * fac * wk.λ)
        @. wk.modal = wk.projection * wk.e
        mul!(wk.coeff_y, wk.Ak, wk.modal)
        mul!(wk.f, wk.By, wk.coeff_y)
        _add_tail!(wk.f, wk.tail, t64, Kk, fac)

        if wk.projection_o !== nothing
            @. wk.e_o = exp(-t64 * wk.λ_o)
            @. wk.modal_o = wk.projection_o * wk.e_o
            mul!(wk.coeff_y_o, wk.Bk, wk.modal_o)
            mul!(wk.f_o, wk.By_o, wk.coeff_y_o)
            _add_tail!(wk.f_o, wk.tail_o, t64, Kk, fac)
        end

        phase0 = Ccoeffs[k] * cis(Kk * x0)
        col = @view spectrum[:, k]
        if _irfft_self_conjugate_mode(k, nx)
            if wk.projection_o !== nothing
                @inbounds @simd for iy in 1:ny
                    col[iy] = nx * real(ComplexF64((wk.f[iy] + wk.f_o[iy]) * phase0))
                end
            else
                @inbounds @simd for iy in 1:ny
                    col[iy] = nx * real(ComplexF64(wk.f[iy] * phase0))
                end
            end
        else
            scale = nx / 2
            if wk.projection_o !== nothing
                @inbounds @simd for iy in 1:ny
                    col[iy] = scale * ComplexF64((wk.f[iy] + wk.f_o[iy]) * phase0)
                end
            else
                @inbounds @simd for iy in 1:ny
                    col[iy] = scale * ComplexF64(wk.f[iy] * phase0)
                end
            end
        end
    end
    @views T[it, :, :] .= irfft(spectrum, nx, 2)[1:_ny, :]
    return nothing
end


function solve_explicit(dom, basis, fy, Ccoeffs, tvals, Pe;
                        show_progress::Bool=true,
                        parallel::Bool=true)

    if basis.even_extension
        fac = 0.25
        Nγ = 2*Nγ
        nfy = vcat(fy, fy)
        Nγ = length(nfy)÷2
        _y = vcat(dom.y, pi .+ dom.y)
        χ_0, χ_e, γ_o = normal_mode_decomposition(nfy, _y; Lx=2π)
        ny = length(_y)
        dx = dom.x[2]-dom.x[1]
        _L = dom.x[end]-dom.x[1] + dx
        _dom = domain(_L, 1, length(dom.x), ny)
    else
        fac = 1.0
        χ_0, χ_e, γ_o = normal_mode_decomposition(fy, 2*dom.y; Lx=2π)
        ny = length(dom.y)
        Nγ = ny÷2
        _dom = dom
    end

    γ = vcat(χ_0, χ_e)[1:Nγ]
    γ_o = γ_o[1:Nγ]

    nt = length(tvals)
    nx = length(dom.x)
    nk = length(dom.K)

    T = zeros(Float64, nt, length(dom.y), nx)

    x0 = dom.x[1]
    any(γ_o .> 1e-6) && _require_basis_odd_coefficients(basis)

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
                    local_work = _prepare_coefficient_mode_work(_dom, basis, γ, γ_o, Pe)
                    local_spectrum = zeros(ComplexF64, ny, nx ÷ 2 + 1)

                    for it in chunk
                        _solve_explicit_time!(T, it, tbuf[it],
                                                           local_spectrum,
                                                           local_work, _dom,
                                                           Ccoeffs, x0,
                                                           nx, ny, nk, fac)
                        advance_progress === nothing || advance_progress()
                    end
                end
            end
            _fetch_all_tasks(tasks)
        else
            work = _prepare_coefficient_mode_work(_dom, basis, γ, γ_o, Pe)
            spectrum = zeros(ComplexF64, ny, nx ÷ 2 + 1)

            for (it, t) in enumerate(tvals)
                _solve_explicit_time!(T, it, t, spectrum,
                                                   work, _dom, Ccoeffs, x0,
                                                   nx, ny, nk, fac)
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


function prepare_explicit_avg(dom, U::AbstractVector, fy, Pe, Ns;
                              precision_policy=:float64, qc1::Union{Nothing,Real}=nothing,
                              qc2::Union{Nothing,Real}=nothing,
                              parallel::Bool=true,
                              show_progress::Bool=true, 
                              sweep_chunk_size::Union{Nothing,Integer}=nothing)
    Peabs = abs(Pe)

    _, β0, β, even_extension = flow_decomposition(U, dom)

    if even_extension
        fac = 0.25
        nfy = vcat(fy,fy)
        ny = vcat(dom.y, pi .+ dom.y)
        χ_0, χ_e, _ = normal_mode_decomposition(nfy, ny; Lx=2π)
        γ = vcat(χ_0, χ_e) 
        Nmax = length(γ)
    else
        fac = 1.0
        χ_0, χ_e, _ = normal_mode_decomposition(fy, 2*dom.y; Lx=2π)
        γ = vcat(χ_0, χ_e) 
        Nmax = length(γ)÷2 # arbitrary but O(size of y)
    end

    data = _collect_spectral_data(dom, Peabs, Ns, β;
                                  Nmax=Nmax,
                                  precision_policy=precision_policy,
                                  qc1=qc1, qc2=qc2,
                                  parallel=parallel,
                                  show_progress=show_progress,
                                  compute_odd=false, 
                                  sweep_chunk_size=sweep_chunk_size, 
                                  even_extension=even_extension) # this is not true always

    work = prepare_mode_data_avg(dom, data.Ak, data.an, γ, β0, Pe)

    return (; β0, work, fac)
end


function prepare_explicit_avg(dom, flow::Symbol, fy, Pe, Ns;
                              precision_policy=:float64, qc1::Union{Nothing,Real}=nothing,
                              qc2::Union{Nothing,Real}=nothing,
                              parallel::Bool=true,
                              show_progress::Bool=true, 
                              sweep_chunk_size::Union{Nothing,Integer}=nothing)

    Peabs = abs(Pe)
    _, β0, β, even_extension = flow_decomposition(flow, dom)
    if even_extension
        fac = 0.25
        nfy = vcat(fy,fy)
        ny = vcat(dom.y, pi .+ dom.y)
        χ_0, χ_e, _ = normal_mode_decomposition(nfy, ny; Lx=2π)
        γ = vcat(χ_0, χ_e) 
        Nmax = length(γ)
    else
        fac = 1.0
        χ_0, χ_e, _ = normal_mode_decomposition(fy, 2*dom.y; Lx=2π)
        γ = vcat(χ_0, χ_e) 
        Nmax = length(γ)÷2 # arbitrary but O(size of y)
    end


    data = _collect_spectral_data(dom, Peabs, Ns, β;
                                  Nmax=Nmax,
                                  precision_policy=precision_policy,
                                  qc1=qc1, qc2=qc2,
                                  parallel=parallel,
                                  show_progress=show_progress,
                                  compute_odd=false, 
                                  sweep_chunk_size=sweep_chunk_size,
                                  even_extension=even_extension)

    work = prepare_mode_data_avg(dom, data.Ak, data.an, γ, β0, Pe)

    return (; β0, work, fac)
end



function _avg_work_scratch(work)
    return [AvgModeWork(w.A0, w.λ, w.γproj, similar(w.e)) for w in work]
end


function _evolve_explicit_avg_time!(T, it, t, spec, work, dom, Ccoeffs, x0, nx, fac)
    fill!(spec, 0)

    for k in eachindex(dom.K)
        wk = work[k]

        @. wk.e = exp(-t * fac * wk.λ )

        # Modal averages may use high precision;
        # FFTW reconstruction is intentionally Float64.
        favg = zero(eltype(wk.e))
        @inbounds @simd for n in eachindex(wk.e)
            favg += wk.A0[n] * wk.e[n] * wk.γproj[n]
        end

        # Big/complex arithmetic may happen above; FFTW boundary is here.
        z = ComplexF64(favg * Ccoeffs[k] * cis(2π * dom.K[k] * x0))

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

    # scalar factor associated with an even periodic extension
    fac = prep.fac 
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
                                                   local_work, dom, Ccoeffs, x0, nx, fac)
                        advance_progress === nothing || advance_progress()
                    end
                end
            end

            _fetch_all_tasks(tasks)
        else
            for (it, t) in enumerate(tvals)
                _evolve_explicit_avg_time!(T, it, t, spec, prep.work, dom,
                                           Ccoeffs, x0, nx, fac)
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
