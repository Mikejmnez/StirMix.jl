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
    for r in 0:(R - 1)
        @inbounds for iy in eachindex(y)
            B[iy, r + 1] = cos(2r * y[iy])
        end
    end
    return B
end

function _sine_y_basis(y::AbstractVector, R::Integer)
    B = Matrix{Float64}(undef, length(y), R)
    for r in 0:(R - 1)
        @inbounds for iy in eachindex(y)
            B[iy, r + 1] = sin(2(r + 1) * y[iy])
        end
    end
    return B
end

function _prepare_coefficient_mode_work_point(dom, basis, γ, γ_o, Pe, k, β0)
    conj_spectrum = Pe < 0

    Ak0 = basis.Ak[k]
    Ak = conj_spectrum ? conj.(Ak0) : Ak0

    R, Nn = size(Ak)
    Kk = 2π * dom.K[k]
    if β0 === nothing
        βshift = im * (Float64(basis.β0) / 2) * Kk * Float64(Pe)
    else
        βshift = im * (Float64(β0) / 2) * Kk * Float64(Pe)
    end

    λ = ComplexF64.(basis.an[k])
    conj_spectrum && conj!(λ)

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

    By = _cosine_y_basis(_ytilde(dom), R)
    Tfield = promote_type(eltype(By), eltype(projection))

    e = Vector{ComplexF64}(undef, Nn)
    modal = similar(projection)
    coeff_y = Vector{Tproj}(undef, R)
    f = Vector{Tfield}(undef, length(dom.y))

    tail = cosine_tail_data(_ytilde(dom), γ, Nn)

    Bk = nothing
    λ_o = nothing
    projection_o = nothing
    By_o = nothing
    e_o = nothing
    modal_o = nothing
    coeff_y_o = nothing
    f_o = nothing
    tail_o = nothing

    if any(x -> abs(x) > 1e-6, γ_o)
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

        By_o = _sine_y_basis(_ytilde(dom), R_o)
        Tfield_o = promote_type(eltype(By_o), eltype(projection_o))

        e_o = Vector{ComplexF64}(undef, Nno)
        modal_o = similar(projection_o)
        coeff_y_o = Vector{Tproj_o}(undef, R_o)
        f_o = Vector{Tfield_o}(undef, length(dom.y))

        tail_o = sine_tail_data(_ytilde(dom), γ_o, R_o)
    end

    return (Ak = Ak, λ = λ, projection = projection, By = By, e = e,
        modal = modal, coeff_y = coeff_y, f = f, tail = tail,
        Bk = Bk, λ_o = λ_o, projection_o = projection_o, By_o = By_o,
        e_o = e_o, modal_o = modal_o, coeff_y_o = coeff_y_o, f_o = f_o,
        tail_o = tail_o)
end

function _prepare_coefficient_mode_work(
        dom::Domain, basis::SpectralBasis, γ::Vector{<:Number},
        γ_o::Vector{<:Number}, Pe::Real, β0::Union{Nothing, Real})
    any(x -> abs(x) > 1e-6, γ_o) && _require_basis_odd_coefficients(basis)

    nk = length(dom.K)
    work = Vector{Any}(undef, nk)

    for k in 1:nk
        work[k] = _prepare_coefficient_mode_work_point(dom, basis, γ, γ_o, Pe, k, β0)
    end

    return work
end

function _prepare_coefficient_mode_work(
        dom::Domain, basis::SpectralBasis, γ::Matrix{ComplexF64},
        γ_o::Matrix{ComplexF64}, Pe::Real, β0::Union{Nothing, Real})
    any(x -> abs(x) > 1e-6, γ_o) && _require_basis_odd_coefficients(basis)

    nk = length(dom.K)
    work = Vector{Any}(undef, nk)

    for k in 1:nk
        work[k] = _prepare_coefficient_mode_work_point(
            dom, basis, γ[:, k], γ_o[:, k], Pe, k, β0)
    end

    return work
end

function _prepare_coefficient_mode_work(
        dom::Domain, basis::SpectralBasis, γ::Vector{<:Number},
        γ_o::Vector{<:Number}, Pe::Real, β0::Union{Nothing, Real}, ik::Integer)
    any(x -> abs(x) > 1e-6, γ_o) && _require_basis_odd_coefficients(basis)

    nk = length(dom.K)
    work = Vector{Any}(undef, nk)
    work[ik] = _prepare_coefficient_mode_work_point(dom, basis, γ, γ_o, Pe, ik, β0)

    return work
end

function _prepare_coefficient_mode_work(
        dom::Domain, basis::SpectralBasis, γ::Matrix{ComplexF64},
        γ_o::Matrix{ComplexF64}, Pe::Real, β0::Union{Nothing, Real}, ik::Integer)
    any(x -> abs(x) > 1e-6, γ_o) && _require_basis_odd_coefficients(basis)

    nk = length(dom.K)
    work = Vector{Any}(undef, nk)
    work[ik] = _prepare_coefficient_mode_work_point(
        dom, basis, γ[:, ik], γ_o[:, ik], Pe, ik, β0)

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

function prepare_mode_data_avg(dom::Domain, Ak::Vector{Any}, avals::Vector{Any},
        γ::Vector{Float64}, β0::Union{Nothing, Real}, Pe::Real)
    conj_spectrum = Pe < 0
    nk = length(dom.K)
    work = Vector{AvgModeWork}(undef, nk)

    βshift = im * (β0 / 2) * Pe

    for k in 1:nk
        Ak0 = Ak[k]
        Akk = conj_spectrum ? conj.(Ak0) : Ak0

        ank = avals[k]
        conj_spectrum && conj!(ank)

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

function prepare_mode_data_avg(dom::Domain, Ak::Vector{Any}, avals::Vector{Any},
        γ::Matrix{ComplexF64}, β0::Union{Nothing, Real}, Pe::Real)
    conj_spectrum = Pe < 0
    nk = length(dom.K)
    work = Vector{AvgModeWork}(undef, nk)

    βshift = im * (β0 / 2) * Pe

    for k in 1:nk
        Ak0 = Ak[k]
        Akk = conj_spectrum ? conj.(Ak0) : Ak0

        ank = avals[k]
        conj_spectrum && conj!(ank)

        Nn = length(ank)
        Kk = 2π * dom.K[k]
        Tmode = promote_type(eltype(Akk), eltype(ank), eltype(γ),
            typeof(βshift * Kk), typeof(Kk^2))

        A0 = Vector{Tmode}(undef, Nn)
        γproj = Vector{Tmode}(undef, Nn)
        e = Vector{Tmode}(undef, Nn)
        λ = Vector{Tmode}(undef, Nn)

        Nγ = min(size(γ, 1), size(Akk, 1))

        copyto!(A0, vec(Akk[1, :]))

        mul!(γproj, transpose(view(Akk, 1:Nγ, :)), view(γ, 1:Nγ, k))

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

function _solve_explicit_mode!(spectrum, work, dom, Ccoeffs, x0, nx, ny, fac, t64, k;
        spectrum_star = nothing)
    wk = work[k]
    Kk = 2π * dom.K[k]

    @. wk.e = exp(-t64 * fac * wk.λ)
    @. wk.modal = wk.projection * wk.e
    mul!(wk.coeff_y, wk.Ak, wk.modal)
    mul!(wk.f, wk.By, wk.coeff_y)
    _add_tail!(wk.f, wk.tail, t64, Kk, fac)

    if wk.projection_o !== nothing
        @. wk.e_o = exp(-t64 * fac * wk.λ_o)
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

    if spectrum_star !== nothing
        col_star = @view spectrum_star[:, k]
        if _irfft_self_conjugate_mode(k, nx)
            if wk.projection_o !== nothing
                @inbounds @simd for iy in 1:ny
                    col_star[iy] = real(ComplexF64(wk.f[iy] + wk.f_o[iy]))
                end
            else
                @inbounds @simd for iy in 1:ny
                    col_star[iy] = real(ComplexF64(wk.f[iy]))
                end
            end
        else
            if wk.projection_o !== nothing
                @inbounds @simd for iy in 1:ny
                    col_star[iy] = ComplexF64(wk.f[iy] + wk.f_o[iy])
                end
            else
                @inbounds @simd for iy in 1:ny
                    col_star[iy] = ComplexF64(wk.f[iy])
                end
            end
        end
    end

    return nothing
end

function _solve_explicit_time!(T, it, t, spectrum, field, xplan, work,
        dom, Ccoeffs, x0, nx, ny, nk, fac;
        parallel::Bool = false,
        modal_k::Union{Nothing, Int} = nothing,
        spectrum_star = nothing)
    fill!(spectrum, 0)
    spectrum_star === nothing || fill!(spectrum_star, 0)
    t64 = Float64(t)
    _ny = size(T, 2)

    if modal_k !== nothing
        _solve_explicit_mode!(spectrum, work, dom, Ccoeffs, x0,
            nx, ny, fac, t64, modal_k;
            spectrum_star = spectrum_star)
    elseif parallel && Threads.nthreads() > 1 && nk > 1
        Threads.@threads :dynamic for k in 1:nk
            _solve_explicit_mode!(spectrum, work, dom, Ccoeffs, x0,
                nx, ny, fac, t64, k;
                spectrum_star = spectrum_star)
        end
    else
        for k in 1:nk
            _solve_explicit_mode!(spectrum, work, dom, Ccoeffs, x0,
                nx, ny, fac, t64, k;
                spectrum_star = spectrum_star)
        end
    end

    mul!(field, xplan, spectrum)
    @views T[it, :, :] .= field[1:_ny, :]
    return nothing
end

_collect_tvals(tvals::Vector{Float64}) = tvals
_collect_tvals(tvals::Real) = [Float64(tvals)]
_collect_tvals(tvals) = collect(Float64, tvals)

function solve_explicit(dom::Domain, basis::SpectralBasis, fy::Vector{Float64},
        Ccoeffs::Vector{<:Number}, tvals;
        β0::Union{Nothing, Real} = nothing,
        Pe::Union{Nothing, Real} = nothing,
        show_progress::Bool = true,
        parallel::Bool = true,
        modal::Bool = false)
    γ, γ_o = y_modes(fy, dom; extend = basis.even_extension)
    if basis.even_extension
        fac = 0.5
        ny = 2 * length(dom.y)
        # same aspect ratio, doubled y grid; dom.L, not the grid period
        _dom = domain(dom.L, dom.M, length(dom.x), ny)
    else
        fac = 1.0
        ny = length(dom.y)
        _dom = dom
    end

    _Pe = something(Pe, basis.Pe)

    nt = length(tvals)
    nx = length(dom.x)
    nk = length(dom.K)

    T = zeros(Float64, nt, length(dom.y), nx)

    x0 = dom.x[1]
    any(x -> abs(x) > 1e-6, γ_o) && _require_basis_odd_coefficients(basis)

    # define progressmeter instance
    p = show_progress && nt > 0 ?
        Progress(nt; desc = "Computing time-evolving solution") :
        nothing

    processed = Ref(0)

    advance_progress = p === nothing ? nothing : () -> begin
        processed[] += 1
        update!(p, processed[])
        return nothing
    end

    completed = false
    old_fft_threads = nothing

    try
        if parallel && !modal && Threads.nthreads() > 1
            old_fft_threads = FFTW.get_num_threads()
            FFTW.set_num_threads(Threads.nthreads())
        end

        ik = modal ? findfirst(!iszero, Ccoeffs) : nothing
        work = ik === nothing ?
               _prepare_coefficient_mode_work(_dom, basis, γ, γ_o, _Pe, β0) :
               _prepare_coefficient_mode_work(_dom, basis, γ, γ_o, _Pe, β0, ik)
        spectrum = zeros(ComplexF64, ny, nx ÷ 2 + 1)
        field = Matrix{Float64}(undef, ny, nx)
        xplan = plan_irfft(spectrum, nx, 2)

        for (it, t) in enumerate(tvals)
            _solve_explicit_time!(T, it, t, spectrum, field, xplan,
                work, _dom, Ccoeffs, x0,
                nx, ny, nk, fac;
                parallel = parallel && !modal,
                modal_k = ik)
            advance_progress === nothing || advance_progress()
        end
        completed = true
    finally
        old_fft_threads === nothing || FFTW.set_num_threads(old_fft_threads)

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

function solve_explicit(dom::Domain, basis::SpectralBasis, fy::Vector{ComplexF64},
        Ccoeffs::Vector{<:Number}, tvals;
        β0::Union{Nothing, Real} = nothing,
        Pe::Union{Nothing, Real} = nothing,
        show_progress::Bool = true,
        parallel::Bool = true,
        modal::Bool = false,
        compute_Φky::Bool = false)
    γ, γ_o = y_modes(fy, dom; extend = basis.even_extension)
    if basis.even_extension
        fac = 0.5
        ny = 2 * length(dom.y)
        # same aspect ratio, doubled y grid; dom.L, not the grid period 
        _dom = domain(dom.L, dom.M, length(dom.x), ny)
    else
        fac = 1.0
        ny = length(dom.y)
        _dom = dom
    end

    _Pe = something(Pe, basis.Pe)

    nt = length(tvals)
    nx = length(dom.x)
    nk = length(dom.K)

    T = zeros(Float64, nt, length(dom.y), nx)

    x0 = dom.x[1]
    any(x -> abs(x) > 1e-6, γ_o) && _require_basis_odd_coefficients(basis)

    # define progressmeter instance
    p = show_progress && nt > 0 ?
        Progress(nt; desc = "Computing time-evolving solution") :
        nothing

    processed = Ref(0)

    advance_progress = p === nothing ? nothing : () -> begin
        processed[] += 1
        update!(p, processed[])
        return nothing
    end

    completed = false
    old_fft_threads = nothing
    spectrum_star = compute_Φky ? zeros(ComplexF64, ny, nx ÷ 2 + 1) : nothing

    try
        if parallel && !modal && Threads.nthreads() > 1
            old_fft_threads = FFTW.get_num_threads()
            FFTW.set_num_threads(Threads.nthreads())
        end

        ik = modal ? findfirst(!iszero, Ccoeffs) : nothing
        work = ik === nothing ?
               _prepare_coefficient_mode_work(_dom, basis, γ, γ_o, _Pe, β0) :
               _prepare_coefficient_mode_work(_dom, basis, γ, γ_o, _Pe, β0, ik)
        spectrum = zeros(ComplexF64, ny, nx ÷ 2 + 1)
        field = Matrix{Float64}(undef, ny, nx)
        xplan = plan_irfft(spectrum, nx, 2)

        for (it, t) in enumerate(tvals)
            _solve_explicit_time!(T, it, t, spectrum, field, xplan,
                work, _dom, Ccoeffs, x0,
                nx, ny, nk, fac;
                parallel = parallel && !modal,
                modal_k = ik,
                spectrum_star = spectrum_star)
            advance_progress === nothing || advance_progress()
        end
        completed = true
    finally
        old_fft_threads === nothing || FFTW.set_num_threads(old_fft_threads)

        if p !== nothing
            if completed
                finish!(p)
            else
                cancel(p, "Time evolution aborted")
            end
        end
    end
    return (T, spectrum_star)
end

function solve_explicit(dom::Domain, basis::SpectralBasis,
        T0::Matrix{Float64}, tvals;
        β0::Union{Nothing, Real} = nothing,
        Pe::Union{Nothing, Real} = nothing,
        show_progress::Bool = true,
        parallel::Bool = true,
        compute_Φky::Bool = false)
    # period of the uniform x grid; valid for any offset of x[1]
    # (this is 2π·dom.L, not dom.L )
    x_period = dom.x[end] - dom.x[1] + (dom.x[2] - dom.x[1])
    # x-decomposition first; the y-extension (if any) acts row-independently, so it
    # is applied afterwards to the coefficients
    c0_2D, ce_2D, co_2D = normal_mode_decomposition(T0, dom.x; Lx = x_period, dim = 2)
    Ces = hcat(c0_2D ./ 2, complex_x_coeffs(ce_2D, co_2D))
    γ, γ_o = y_modes(Ces, dom; extend = basis.even_extension)
    if basis.even_extension
        fac = 0.5
        ny = 2 * length(dom.y)
        # same aspect ratio, doubled y grid; dom.L, not the grid period
        _dom = domain(dom.L, dom.M, length(dom.x), ny)
    else
        fac = 1.0
        ny = length(dom.y)
        _dom = dom
    end

    # in the following, each Fourier coeff has y dependence
    _Pe = something(Pe, basis.Pe)
    # identify to reuse function
    Ccoeffs = ones(ComplexF64, length(dom.K))

    nt, nx, nk = length(tvals), length(dom.x), length(dom.K)

    T = zeros(Float64, nt, length(dom.y), nx)

    x0 = dom.x[1]
    any(x -> abs(x) > 1e-6, γ_o) && _require_basis_odd_coefficients(basis)

    # define progressmeter instance
    p = show_progress && nt > 0 ?
        Progress(nt; desc = "Computing time-evolving solution") :
        nothing

    processed = Ref(0)

    advance_progress = p === nothing ? nothing : () -> begin
        processed[] += 1
        update!(p, processed[])
        return nothing
    end

    completed = false
    old_fft_threads = nothing

    spectrum_star = compute_Φky ? zeros(ComplexF64, ny, nx ÷ 2 + 1) : nothing

    try
        if parallel && Threads.nthreads() > 1
            old_fft_threads = FFTW.get_num_threads()
            FFTW.set_num_threads(Threads.nthreads())
        end

        work = _prepare_coefficient_mode_work(_dom, basis, γ, γ_o, _Pe, β0)
        spectrum = zeros(ComplexF64, ny, nx ÷ 2 + 1)
        field = Matrix{Float64}(undef, ny, nx)
        # precompute the plan for the inverse transform in x
        xplan = plan_irfft(spectrum, nx, 2)

        for (it, t) in enumerate(tvals)
            _solve_explicit_time!(T, it, t, spectrum, field, xplan,
                work, _dom, Ccoeffs, x0,
                nx, ny, nk, fac;
                parallel = parallel,
                spectrum_star = spectrum_star)
            advance_progress === nothing || advance_progress()
        end
        completed = true
    finally
        old_fft_threads === nothing || FFTW.set_num_threads(old_fft_threads)

        if p !== nothing
            if completed
                finish!(p)
            else
                cancel(p, "Time evolution aborted")
            end
        end
    end
    return compute_Φky ? (T, spectrum_star) : T
end

function solve_explicit(dom::Domain, bases::BasesIter,
        T0::Matrix{Float64}, tvals;
        β0::Union{Nothing, Real} = nothing,
        show_progress::Bool = true)

    # discretize time evolution
    indT = bases.indT
    ivals = bases.ivals
    PEs = bases.Pes
    basis = bases.basis
    τ = tvals
    T = zeros(Float64, length(tvals), length(dom.y), length(dom.x))

    p = show_progress && !isempty(indT) ?
        Progress(length(indT); desc = "Computing time-evolving solution") :
        nothing
    t0 = 0.0
    for it in eachindex(indT)
        i1, i2 = indT[it]
        tau = range(start = τ[i1], step = step(τ), stop = τ[i2]) .- t0
        Tn = solve_explicit(dom, basis[ivals[i1]], T0, tau; show_progress = false,
            parallel = true, β0 = β0, Pe = PEs[i1])
        T[i1:i2, :, :] = Tn
        T0 = Tn[end, :, :]
        t0 = τ[i2]
        p === nothing || next!(p)
    end

    p === nothing || finish!(p)
    return T
end

function _avg_work_scratch(work)
    return [AvgModeWork(w.A0, w.λ, w.γproj, similar(w.e)) for w in work]
end

function _evolve_explicit_avg_time!(T, it, t, spec, work, dom, Ccoeffs, x0, nx, fac)
    fill!(spec, 0)

    # it is missing the tail - initial condition that decay are pure normal mode
    # soln to diffusion equation (see add_tail! function in case of non-avg soln)

    for k in eachindex(dom.K)
        wk = work[k]

        @. wk.e = exp(-t * fac * wk.λ)

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

function evolve_explicit_avg(dom::Domain, basis::SpectralBasis,
        fy::Vector{<:Number}, Ccoeffs::Vector{<:Number}, tvals;
        β0::Union{Nothing, Real} = nothing,
        Pe::Union{Nothing, Real} = nothing,
        show_progress::Bool = true,
        parallel::Bool = true)
    # the averaged path uses only the cosine (even) modes
    γ, _ = y_modes(fy, dom; extend = basis.even_extension)
    fac = basis.even_extension ? 0.5 : 1.0

    nt = length(tvals)
    nx = length(dom.x)

    T = zeros(Float64, nt, nx)
    spec = zeros(ComplexF64, nx ÷ 2 + 1)

    x0 = dom.x[1]

    _Pe = something(Pe, basis.Pe)
    _β0 = something(β0, basis.β0)
    work = prepare_mode_data_avg(dom, basis.Ak, basis.an, γ, _β0, _Pe)

    # define progressmeter instance
    p = show_progress && nt > 0 ?
        Progress(nt; desc = "Computing time-evolving solution") :
        nothing

    processed = Ref(0)

    advance_progress = p === nothing ? nothing : () -> begin
        processed[] += 1
        update!(p, processed[])
        return nothing
    end

    completed = false

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
                    local_work = _avg_work_scratch(work)
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
                _evolve_explicit_avg_time!(T, it, t, spec, work, dom,
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

function evolve_explicit_avg(dom::Domain, basis::SpectralBasis,
        fy::Matrix{ComplexF64}, Ccoeffs::Vector{<:Number}, tvals;
        parallel::Bool = true,
        show_progress::Bool = true,
        β0::Union{Nothing, Real} = nothing,
        Pe::Union{Nothing, Real} = nothing)
    # the averaged path uses only the cosine (even) modes
    γ, _ = y_modes(fy, dom; extend = basis.even_extension)
    fac = basis.even_extension ? 0.5 : 1.0

    nt = length(tvals)
    nx = length(dom.x)

    T = zeros(Float64, nt, nx)
    spec = zeros(ComplexF64, nx ÷ 2 + 1)

    x0 = dom.x[1]

    _Pe = something(Pe, basis.Pe)
    _β0 = something(β0, basis.β0)
    work = prepare_mode_data_avg(dom, basis.Ak, basis.an, γ, _β0, _Pe)

    for (it, t) in enumerate(tvals)
        _evolve_explicit_avg_time!(T, it, t, spec, work, dom,
            Ccoeffs, x0, nx, fac)
    end

    return T
end

function evolve_explicit_avg(dom::Domain, bases::BasesIter,
        fy::Vector{Float64}, Ccoeffs::Vector{<:Number}, tvals;
        β0::Union{Nothing, Real} = nothing,
        show_progress::Bool = true)

    # discretize time evolution
    indT = bases.indT
    ivals = bases.ivals
    PEs = bases.Pes
    basis = bases.basis

    τ = tvals
    T = zeros(Float64, length(tvals), length(dom.x))

    p = show_progress && !isempty(indT) ?
        Progress(length(indT); desc = "Computing time-evolving solution") :
        nothing
    t0 = 0.0
    fx = reconstruct_from_complex_coeffs(dom.x, 2pi * dom.K, Ccoeffs)
    Tp = fy * transpose(fx)
    C = Ccoeffs

    for it in eachindex(indT)
        tau = range(start = τ[indT[it][1]], step = step(τ), stop = τ[indT[it][2]]) .- t0
        Tn = evolve_explicit_avg(dom, basis[ivals[indT[it][1]]], fy, C, tau;
            show_progress = false, parallel = true, β0 = β0, Pe = PEs[indT[it][1]])
        T[indT[it][1]:indT[it][2], :] = Tn
        t0 = τ[indT[it][2]]
        # get 2d-spectrum - the x-coefficients are already applied in Tp, don't apply them again
        Tp, fy = solve_explicit(
            dom, basis[ivals[indT[it][1]]], Tp, [tau[1], tau[end]]; show_progress = false,
            parallel = true, β0 = β0, Pe = PEs[indT[it][1]], compute_Φky = true)
        # For even-extension flows the spectrum lives on the doubled y-grid.
        # keep only the lower half.
        if basis[ivals[indT[it][1]]].even_extension
            fy = fy[1:length(dom.y), :]
        end
        Tp = Tp[end, :, :]
        C = ones(ComplexF64, length(dom.K))
        p === nothing || next!(p)
    end

    p === nothing || finish!(p)
    return T
end
