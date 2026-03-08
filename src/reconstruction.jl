
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
    @assert length(a) == length(b) "a and b must have the same length"
    K = length(a)
    ω0 = 2π / Lx
    f = fill(a0/2, length(x))
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