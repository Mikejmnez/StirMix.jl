
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