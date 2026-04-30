using StirMix
using Test

@testset "StirMix.jl" begin
    dom = domain(1.0, 1, 8, 8)

    @testset "spectral basis eigenfunction toggle" begin
        basis = spectral_basis(:cos, 1.0, dom, 10)
        @test length(basis.qs) == length(dom.K)
        @test basis.Φe !== nothing
        @test basis.Φo !== nothing
        @test all(x -> x !== nothing, basis.Φe)
        @test all(x -> x !== nothing, basis.Φo)

        mixed_basis = spectral_basis(:cos, 10000.0, dom, 10;
                                     compute_eigenfunctions=false)
        @test mixed_basis.Φe === nothing
        @test mixed_basis.Φo === nothing
        @test mixed_basis.bn !== nothing
        @test mixed_basis.Bk !== nothing
        @test any(q -> q isa ComplexF64, mixed_basis.qs)
        @test any(q -> q isa Complex{BigFloat}, mixed_basis.qs)
    end

    @testset "averaged preparation reuses mixed spectral data" begin
        gamma = ones(5)
        ccoeffs = ones(ComplexF64, length(dom.K))

        basis = spectral_basis(:cos, 1.0, dom, 10)
        field = solve_explicit(dom, basis, gamma, ccoeffs, [0.1], 1.0)
        @test size(field) == (1, length(dom.y), length(dom.x))

        prep = prepare_explicit_avg(dom, :cos, gamma, 10000.0, 10)
        avg_field = evolve_explicit_avg(dom, prep, ccoeffs, [0.1])

        @test prep.β0 == 1.0
        @test length(prep.work) == length(dom.K)
        @test any(w -> eltype(w.e) === Complex{BigFloat}, prep.work)
        @test size(avg_field) == (1, length(dom.x))
    end
end
