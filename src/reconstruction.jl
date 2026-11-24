
"""
    reconstruct!(
        f::AbstractVector{Tf},
        U::AbstractMatrix{Tu},
        γ::AbstractVector{Tγ},
        COStail::AbstractMatrix{Tb},
        γtail::AbstractVector{Tγt},
        m_tail::AbstractVector{<:Integer},
        tmp::AbstractVector{Ttmp},
        t::Tt,
        Kk::Tk,
    ) where {Tf,Tu,Tγ,Tb,Tγt,Ttmp,Tt<:Real,Tk<:Real}

Reconstruct a 1D field `f(y,t)` from a hybrid spectral expansion consisting of:

1. a low-mode "gravest" part stored in the columns of `U`, and
2. a higher-mode cosine tail stored in `COStail`, with exponential decay.

The reconstruction is

```math
f(y,t) =
\\sum_{m=0}^{N_k-1} \\gamma_m \\, U_m(y,t)
+
\\sum_{m=N_k}^{M_{\\max}}
\\gamma_m \\, e^{-t(m^2 + K_k^2)} \\, \\cos(my),
"""
function reconstruct!(f, U, γ, COStail, γtail, m_tail, tmp, t, Kk)
    fill!(f, zero(eltype(f)))
    Nuse = min(size(U, 2), length(γ))
    # Gravest part
    if Nuse > 0
        mul!(@view(f[:]), @view(U[:, 1:Nuse]), @view(γ[1:Nuse]))
    end
    # Tail part
    if !isempty(m_tail)
        k2 = Kk^2
        tmp .= γtail .* exp.(-t .* (m_tail .^ 2 .+ k2))
        f .+= COStail * tmp
    end

    return f
end

"""
    reconstruct_fourier_series(x, a0, a, b; Lx)

Reconstruct f(x) from Fourier series coefficients:

f(x) ≈ a0/2 + Σ_{k=1..K} [ a[k]*cos(2πk x/Lx) + b[k]*sin(2πk x/Lx) ]

Requires length(a) == length(b).
"""
function reconstruct_fourier_series(x::AbstractVector, a0::Real,
        a::AbstractVector, b::AbstractVector; Lx::Real)
    @assert length(a)==length(b) "a and b must have the same length"
    K = length(a)
    ω0 = 2π / Lx
    f = fill(a0 / 2, length(x))
    @inbounds for k in 1:K
        wk = k * ω0
        f .+= a[k] .* cos.(wk .* x) .+ b[k] .* sin.(wk .* x)
    end
    return f
end

function reconstruct_from_complex_coeffs(x, K, C)
    f = zeros(Float64, length(x))
    for k in eachindex(K)
        f .+= real.(C[k] .* exp.(im .* K[k] .* x))
    end
    return f
end

"""
    time_resample(ft::AbstractVector{<:Real}; nt::Integer=0)

Discretize a real-valued periodic signal `ft` into a piecewise-constant signal
with levels spaced by

    Δ = 1 / 2^nt

over the interval [0, 1] when `ft` is nonnegative, and over [-1, 1]
otherwise.

Returns `(nft, vals, ivals, vals_pos, ivals_pos)` where

- `nft`   : discretized signal (retains sign)
- `vals`  : all allowed discrete values, ordered increasingly
- `ivals` : indices such that `vals[ivals[i]] == nft[i]`
- `vals_pos`  : the nonnegative half of `vals` (i.e. `0:Δ:1`) — the distinct
                `|·|` magnitudes present in `nft`
- `ivals_pos` : indices such that `vals_pos[ivals_pos[i]] == abs(nft[i])`

Mathematically,
    `vals[ivals[i]] == nft[i]`
    `vals_pos[ivals_pos[i]] == abs(nft[i])`

When `ft` is already nonnegative, `vals_pos`/`ivals_pos` alias `vals`/`ivals`
directly (no fold needed).

Notes:
- `ivals` and `ivals_pos` use Julia's 1-based indexing.
- This version uses nearest-level rounding
"""
function time_resample(ft::AbstractVector{<:Real}; nt::Integer = 0)
    nt < 0 && throw(ArgumentError("nt must be >= 0"))

    T = float(eltype(ft))
    Δ = one(T) / T(2^nt)

    # compute left (lo) and right (hi) endpoints
    lo = !isempty(ft) && all(x -> T(x) >= zero(T), ft) ? zero(T) : -one(T)
    hi = one(T)

    # vals is monotonic, 
    vals = collect(lo:Δ:hi)

    nft = Vector{T}(undef, length(ft))
    ivals = Vector{Int}(undef, length(ft))

    @inbounds for i in eachindex(ft)
        # nearest allowed level
        y = clamp(Δ * round(T(ft[i]) / Δ), lo, hi)
        nft[i] = y

        # map y to its position in vals
        ivals[i] = Int(round((y - lo) / Δ)) + 1
    end

    # check if there are negative values in ft, if so folder them to 
    # estimate positive vals and ivals indexes.
    if lo < 0
        # vals is symmetric about 0 (odd length), so folding an index onto
        # the nonnegative half is a reflection about the center index.
        center = (length(vals) + 1) ÷ 2
        vals_pos = vals[center:end]
        ivals_pos = [abs(iv - center) + 1 for iv in ivals]
    else
        vals_pos, ivals_pos = vals, ivals
    end

    return nft, vals, ivals, vals_pos, ivals_pos
end

"""
    indt_intervals(ivals::Vector{Integer})
"""
function indt_intervals(ivals)
    indt = Tuple{Int, Int}[]

    t0 = 1

    for k in 1:(length(ivals) - 1)
        if ivals[k] != ivals[k + 1]
            push!(indt, (t0, k))
            t0 = k + 1
        end
    end

    push!(indt, (t0, length(ivals)))

    return indt
end
