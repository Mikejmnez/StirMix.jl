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

    @testset "spectral basis parallel chunks and progress" begin
        serial_basis, serial_output = capture_stderr() do
            spectral_basis(:cos, 1.0, dom, 10;
                           precision_policy=:float64,
                           parallel=false,
                           show_progress=true,
                           sweep_chunk_size=Int32(2))
        end
        parallel_basis, parallel_output = capture_stderr() do
            spectral_basis(:cos, 1.0, dom, 10;
                           precision_policy=:float64,
                           parallel=true,
                           show_progress=true,
                           sweep_chunk_size=2)
        end

        @test serial_basis.qs == parallel_basis.qs
        @test all(k -> serial_basis.an[k] == parallel_basis.an[k], eachindex(dom.K))
        @test all(k -> serial_basis.Ak[k] == parallel_basis.Ak[k], eachindex(dom.K))
        @test all(k -> serial_basis.bn[k] == parallel_basis.bn[k], eachindex(dom.K))
        @test all(k -> serial_basis.Bk[k] == parallel_basis.Bk[k], eachindex(dom.K))
        @test all(k -> serial_basis.Φe[k] == parallel_basis.Φe[k], eachindex(dom.K))
        @test all(k -> serial_basis.Φo[k] == parallel_basis.Φo[k], eachindex(dom.K))
        @test isempty(serial_output) || occursin("Computing spectral basis", serial_output)
        @test isempty(parallel_output) || occursin("Computing spectral basis", parallel_output)
        @test !occursin("aborted", lowercase(serial_output))
        @test !occursin("aborted", lowercase(parallel_output))

        mixed_serial = spectral_basis(:cos, 10000.0, dom, 10;
                                      precision_policy=:mixed,
                                      compute_eigenfunctions=false,
                                      parallel=false,
                                      sweep_chunk_size=2)
        mixed_parallel = spectral_basis(:cos, 10000.0, dom, 10;
                                        precision_policy=:mixed,
                                        compute_eigenfunctions=false,
                                        parallel=true,
                                        sweep_chunk_size=2)

        @test mixed_serial.qs == mixed_parallel.qs
        @test any(q -> q isa Complex{BigFloat}, mixed_parallel.qs)
        @test mixed_parallel.Φe === nothing
        @test mixed_parallel.Φo === nothing
        @test all(k -> mixed_serial.an[k] == mixed_parallel.an[k], eachindex(dom.K))
        @test all(k -> mixed_serial.Ak[k] == mixed_parallel.Ak[k], eachindex(dom.K))
        @test all(k -> mixed_serial.bn[k] == mixed_parallel.bn[k], eachindex(dom.K))
        @test all(k -> mixed_serial.Bk[k] == mixed_parallel.Bk[k], eachindex(dom.K))

        @test_throws ArgumentError spectral_basis(:cos, 1.0, dom, 10;
                                                  precision_policy=:float64,
                                                  sweep_chunk_size=0)
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

    @testset "averaged preparation parallel chunks" begin
        gamma = ones(5)
        U, _, _ = predefined_flow(:cos, dom)

        serial_prep, serial_output = capture_stderr() do
            prepare_explicit_avg(dom, U, gamma, 1.0, 10;
                                 precision_policy=:float64,
                                 parallel=false,
                                 show_progress=true,
                                 sweep_chunk_size=Int32(2))
        end
        parallel_prep, parallel_output = capture_stderr() do
            prepare_explicit_avg(dom, :cos, gamma, 1.0, 10;
                                 precision_policy=:float64,
                                 parallel=true,
                                 show_progress=true,
                                 sweep_chunk_size=2)
        end

        @test length(serial_prep.work) == length(dom.K)
        @test length(parallel_prep.work) == length(dom.K)
        @test serial_prep.β0 == parallel_prep.β0
        @test all(k -> serial_prep.work[k].A0 == parallel_prep.work[k].A0, eachindex(dom.K))
        @test all(k -> serial_prep.work[k].λ == parallel_prep.work[k].λ, eachindex(dom.K))
        @test all(k -> serial_prep.work[k].γproj == parallel_prep.work[k].γproj, eachindex(dom.K))
        @test isempty(serial_output) || occursin("Computing spectral basis", serial_output)
        @test isempty(parallel_output) || occursin("Computing spectral basis", parallel_output)
        @test !occursin("aborted", lowercase(serial_output))
        @test !occursin("aborted", lowercase(parallel_output))
    end

    @testset "explicit evolution progress" begin
        gamma = ones(5)
        ccoeffs = ones(ComplexF64, length(dom.K))
        tvals = [0.1, 0.2, 0.3]
        progress_tvals = ProgressTestTimes(tvals)

        basis = spectral_basis(:cos, 1.0, dom, 10; precision_policy=:float64)
        field = solve_explicit(dom, basis, gamma, ccoeffs, tvals, 1.0)
        parallel_field = solve_explicit(dom, basis, gamma, ccoeffs, tvals, 1.0;
                                        parallel=true)

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
        parallel_avg = evolve_explicit_avg(dom, prep, ccoeffs, tvals;
                                           parallel=true)

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
        @test parallel_field ≈ field
        @test isempty(no_progress_output)
        @test occursin("Computing time-evolving solution", progress_output)
        @test !occursin("aborted", lowercase(progress_output))

        @test no_progress_avg == avg_field
        @test progress_avg == avg_field
        @test parallel_avg ≈ avg_field
        @test isempty(no_progress_avg_output)
        @test occursin("Computing time-evolving solution", progress_avg_output)
        @test !occursin("aborted", lowercase(progress_avg_output))
    end
end
