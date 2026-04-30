using StirMix
using Test

struct ProgressTestTimes{T}
    values::Vector{T}
    delay::Float64
end

ProgressTestTimes(values; delay=0.12) = ProgressTestTimes(collect(values), delay)

Base.length(tvals::ProgressTestTimes) = length(tvals.values)

function Base.iterate(tvals::ProgressTestTimes, state=1)
    state > length(tvals.values) && return nothing
    sleep(tvals.delay)
    return tvals.values[state], state + 1
end

function capture_stderr(f)
    mktemp() do _, io
        result = redirect_stdout(devnull) do
            redirect_stderr(io) do
                f()
            end
        end

        flush(io)
        seekstart(io)
        return result, read(io, String)
    end
end

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

        float64_progress_basis = redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                spectral_basis(:cos, 1.0, dom, 10;
                               precision_policy=:float64,
                               show_progress=true)
            end
        end
        mixed_progress_basis = redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                spectral_basis(:cos, 10000.0, dom, 10;
                               precision_policy=:mixed,
                               compute_eigenfunctions=false,
                               show_progress=true)
            end
        end

        @test length(float64_progress_basis.qs) == length(dom.K)
        @test length(mixed_progress_basis.qs) == length(dom.K)
        @test any(q -> q isa Complex{BigFloat}, mixed_progress_basis.qs)
    end

    @testset "averaged preparation reuses mixed spectral data" begin
        gamma = ones(5)
        ccoeffs = ones(ComplexF64, length(dom.K))

        basis = spectral_basis(:cos, 1.0, dom, 10)
        field = solve_explicit(dom, basis, gamma, ccoeffs, [0.1], 1.0)
        @test size(field) == (1, length(dom.y), length(dom.x))

        prep = prepare_explicit_avg(dom, :cos, gamma, 10000.0, 10)
        avg_field = evolve_explicit_avg(dom, prep, ccoeffs, [0.1])

        U, _, _ = predefined_flow(:cos, dom)
        symbol_progress_prep = redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                prepare_explicit_avg(dom, :cos, gamma, 1.0, 10;
                                     precision_policy=:float64,
                                     show_progress=true)
            end
        end
        vector_progress_prep = redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                prepare_explicit_avg(dom, U, gamma, 1.0, 10;
                                     precision_policy=:float64,
                                     show_progress=true)
            end
        end

        @test prep.β0 == 1.0
        @test length(prep.work) == length(dom.K)
        @test any(w -> eltype(w.e) === Complex{BigFloat}, prep.work)
        @test size(avg_field) == (1, length(dom.x))
        @test length(symbol_progress_prep.work) == length(dom.K)
        @test length(vector_progress_prep.work) == length(dom.K)
    end

    @testset "explicit evolution progress" begin
        gamma = ones(5)
        ccoeffs = ones(ComplexF64, length(dom.K))
        tvals = [0.1, 0.2, 0.3]
        progress_tvals = ProgressTestTimes(tvals)

        basis = spectral_basis(:cos, 1.0, dom, 10; precision_policy=:float64)
        field = solve_explicit(dom, basis, gamma, ccoeffs, tvals, 1.0)

        no_progress_field, no_progress_output = capture_stderr() do
            solve_explicit(dom, basis, gamma, ccoeffs, progress_tvals, 1.0;
                           show_progress=false)
        end
        progress_field, progress_output = capture_stderr() do
            solve_explicit(dom, basis, gamma, ccoeffs, progress_tvals, 1.0;
                           show_progress=true)
        end

        prep = prepare_explicit_avg(dom, :cos, gamma, 1.0, 10;
                                    precision_policy=:float64)
        avg_field = evolve_explicit_avg(dom, prep, ccoeffs, tvals)

        no_progress_avg, no_progress_avg_output = capture_stderr() do
            evolve_explicit_avg(dom, prep, ccoeffs, progress_tvals;
                                show_progress=false)
        end
        progress_avg, progress_avg_output = capture_stderr() do
            evolve_explicit_avg(dom, prep, ccoeffs, progress_tvals;
                                show_progress=true)
        end

        @test no_progress_field == field
        @test progress_field == field
        @test isempty(no_progress_output)
        @test occursin("Computing time-evolving solution", progress_output)
        @test !occursin("aborted", lowercase(progress_output))

        @test no_progress_avg == avg_field
        @test progress_avg == avg_field
        @test isempty(no_progress_avg_output)
        @test occursin("Computing time-evolving solution", progress_avg_output)
        @test !occursin("aborted", lowercase(progress_avg_output))
    end
end
