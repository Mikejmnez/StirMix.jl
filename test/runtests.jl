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
        basis = spectral_basis(:cos, 1.0, dom, 10; 
                               show_progress=false,
                               compute_eigenfunctions=true)
        @test length(basis.qs) == length(dom.K)
        @test basis.Φe !== nothing
        @test basis.Φo === nothing
        @test all(x -> x !== nothing, basis.Φe)

        mixed_basis = spectral_basis(:cos, 1000.0, dom, 10;
                                     show_progress=false,
                                     precision_policy=:mixed, 
                                     qc1=5000, qc2=12000)
        @test mixed_basis.Φe === nothing
        @test mixed_basis.Φo === nothing
        @test mixed_basis.bn === nothing
        @test mixed_basis.Bk === nothing
        @test any(q -> q isa ComplexF64, mixed_basis.qs)
        @test any(q -> q isa Complex{BigFloat}, mixed_basis.qs)

        even_basis = spectral_basis(:cos, 1.0, dom, 10;
                                    show_progress=false)
        @test even_basis.Φe === nothing
        @test even_basis.Φo === nothing
        @test even_basis.bn === nothing
        @test even_basis.Bk === nothing

        float64_progress_basis = redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                spectral_basis(:cos, 1.0, dom, 10;
                               precision_policy=:float64)
            end
        end
        mixed_progress_basis = redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                spectral_basis(:cos, 1000.0, dom, 10;
                               precision_policy=:mixed,
                               qc1=5000, qc2=15000)
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
                           compute_eigenfunctions=true,
                           compute_odd=true,
                           sweep_chunk_size=Int32(2))
        end
        parallel_basis, parallel_output = capture_stderr() do
            spectral_basis(:cos, 1.0, dom, 10;
                           precision_policy=:float64,
                           compute_eigenfunctions=true,
                           compute_odd=true,
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

        mixed_serial = spectral_basis(:cos, 1000.0, dom, 10;
                                      precision_policy=:mixed,
                                      qc1=5000, qc2=15000,
                                      parallel=false,
                                      sweep_chunk_size=2,
                                      show_progress=false,
                                      compute_eigenfunctions=true,
                                      compute_odd=true)
        mixed_parallel = spectral_basis(:cos, 1000.0, dom, 10;
                                        precision_policy=:mixed,
                                        qc1=5000, qc2=15000,
                                        parallel=true,
                                        sweep_chunk_size=2,
                                        show_progress=false,
                                        compute_eigenfunctions=true,
                                        compute_odd=true)

        @test mixed_serial.qs == mixed_parallel.qs
        @test any(q -> q isa Complex{BigFloat}, mixed_parallel.qs)
        @test mixed_parallel.Φe !== nothing
        @test mixed_parallel.Φo !== nothing
        @test all(k -> mixed_serial.an[k] == mixed_parallel.an[k], eachindex(dom.K))
        @test all(k -> mixed_serial.Ak[k] == mixed_parallel.Ak[k], eachindex(dom.K))
        @test all(k -> mixed_serial.bn[k] == mixed_parallel.bn[k], eachindex(dom.K))
        @test all(k -> mixed_serial.Bk[k] == mixed_parallel.Bk[k], eachindex(dom.K))

        @test_throws ArgumentError spectral_basis(:cos, 1.0, dom, 10;
                                                  show_progress=:false,
                                                  precision_policy=:float64,
                                                  sweep_chunk_size=0)
    end

    @testset "averaged preparation reuses mixed spectral data" begin
        gamma = ones(5)
        ccoeffs = ones(ComplexF64, length(dom.K))

        basis = spectral_basis(:cos, 1.0, dom, 10; show_progress=false)
        field = solve_explicit(dom, basis, gamma, nothing, ccoeffs, [0.1], 1.0;
                               show_progress=false)
        @test size(field) == (1, length(dom.y), length(dom.x))

        prep = prepare_explicit_avg(dom, :cos, gamma, 1000.0, 10;
                                    precision_policy=:mixed,
                                    qc1=5000, qc2 = 15000,
                                    show_progress=false)
        avg_field = evolve_explicit_avg(dom, prep, ccoeffs, [0.1]; 
                                        show_progress=false)

        U, _, _ = predefined_flow(:cos, dom)
        symbol_progress_prep = redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                prepare_explicit_avg(dom, :cos, gamma, 1.0, 10;
                                     precision_policy=:float64)
            end
        end
        vector_progress_prep = redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                prepare_explicit_avg(dom, U, gamma, 1.0, 10;
                                     precision_policy=:float64)
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

        basis = spectral_basis(:cos, 1.0, dom, 10; precision_policy=:float64, show_progress=false)
        field = solve_explicit(dom, basis, gamma, nothing, ccoeffs, tvals, 1.0, show_progress=false)
        parallel_field = solve_explicit(dom, basis, gamma, nothing, ccoeffs, tvals, 1.0;
                                        parallel=true, show_progress=false)
        coefficient_field = solve_explicit(dom, basis, gamma, nothing, ccoeffs, tvals, 1.0;
                                           reconstruction=:coefficients, show_progress=false)

        coefficient_basis = spectral_basis(:cos, 1.0, dom, 10;
                                           show_progress=false,
                                           precision_policy=:float64,
                                           compute_eigenfunctions=false,
                                           compute_odd=false)
        coefficient_only_field = solve_explicit(dom, coefficient_basis, gamma, nothing,
                                                ccoeffs, tvals, 1.0;
                                                show_progress=false,
                                                reconstruction=:coefficients)

        no_progress_field, no_progress_output = capture_stderr() do
            solve_explicit(dom, basis, gamma, nothing, ccoeffs, progress_tvals, 1.0;
                           show_progress=false)
        end
        progress_field, progress_output = capture_stderr() do
            solve_explicit(dom, basis, gamma, nothing, ccoeffs, progress_tvals, 1.0;
                           show_progress=true)
        end

        prep = prepare_explicit_avg(dom, :cos, gamma, 1.0, 10;
                                    precision_policy=:float64, show_progress=false)
        avg_field = evolve_explicit_avg(dom, prep, ccoeffs, tvals, show_progress=false)
        parallel_avg = evolve_explicit_avg(dom, prep, ccoeffs, tvals;
                                           parallel=true, show_progress=false)

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
        @test coefficient_field ≈ field
        @test coefficient_only_field ≈ field
        @test coefficient_basis.Φe === nothing
        @test coefficient_basis.bn === nothing
        @test_throws ArgumentError solve_explicit(dom, coefficient_basis, gamma, nothing,
                                                  ccoeffs, tvals, 1.0; 
                                                  reconstruction=:eigenfunctions)
        @test_throws ArgumentError solve_explicit(dom, basis, gamma, ccoeffs, nothing,
                                                  tvals, 1.0;
                                                  show_progress=false,
                                                  reconstruction=:unknown)
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

    @testset "explicit evolution coefficient reconstruction" begin
        gamma = ones(5)
        ccoeffs = ones(ComplexF64, length(dom.K))
        tvals = [0.1]

        basis_with_phi = spectral_basis(:cos, 10.0, dom, 4;
                                        show_progress=false,
                                        precision_policy=:mixed,
                                        qc1=0.0,
                                        qc2=1.0e9,
                                        compute_eigenfunctions=true,
                                        compute_odd=false)
        basis_coeff = spectral_basis(:cos, 10.0, dom, 4;
                                     precision_policy=:mixed,
                                     show_progress=false,
                                     qc1=0.0,
                                     qc2=1.0e9,
                                     compute_eigenfunctions=false,
                                     compute_odd=false)

        field_with_phi = solve_explicit(dom, basis_with_phi, gamma, nothing, ccoeffs,
                                        tvals, 10.0; show_progress=false)
        field_coeff = solve_explicit(dom, basis_coeff, gamma, nothing, ccoeffs,
                                     tvals, 10.0; show_progress=false,
                                     reconstruction=:coefficients)

        @test basis_coeff.Φe === nothing
        @test basis_coeff.Φo === nothing
        @test basis_coeff.bn === nothing
        @test basis_coeff.Bk === nothing
        @test any(k -> eltype(basis_coeff.Ak[k]) === Complex{BigFloat},
                  eachindex(dom.K))
        @test field_coeff ≈ field_with_phi
    end
    @testset "flow decomposition comparison: predetermined vs prescribed" begin
        Nx=8
        Ny=28
        dom = domain(1.0, 1, Nx, Ny)
        U, β0_1, β_1 = predefined_flow(:poiseuille, dom)
        β0_2, β_2, _ = normal_mode_decomposition(U, dom.y; Lx=Ny)
    end

    @testset "tail normal modes" begin 
        Nx = 20
        Ny = 200
        M  = 1
        L  = 1/pi
        dom = domain(L, M, Nx, Ny)

        # ==================================
        # Initial condition in cross-stream direction
        # ==================================
        σ2_y0 = 1/40
        g = @. exp(-(((dom.y-pi/4)/(sqrt(2*σ2_y0))).^2))
        χ_0, χ_e, γ_o = normal_mode_decomposition(g, 2*dom.y; Lx=2π)
        γ_e = vcat(χ_0, χ_e);
        
        NMax = 10 # Tail data begins with the following normal mode
        Nnm = length(γ_e) - NMax # how many tail cosine modes there will be
        NNmo = length(γ_o) - NMax

        m_etail, Atail, γ_etail, _ = cosine_tail_data(dom.y, γ_e, NMax)
        m_otail, Btail, γ_otail, _ = sine_tail_data(dom.y, γ_o, NMax)

        @test size(Atail) == (length(dom.y), Nnm)
        @test γ_etail[1] === γ_e[NMax+1]
        @test γ_otail[1] === γ_o[NMax+1]

        @test isapprox(Atail[:, 1], cos.(NMax*2*dom.y); atol = 1e-4, rtol = 1e-4)
        @test isapprox(Btail[:, 1], sin.(NMax*2*dom.y); atol = 1e-4, rtol = 1e-4)

        @test isapprox(Atail[:, 1], cos.(m_etail[1]*2*dom.y); atol = 1e-4, rtol = 1e-4)
        @test isapprox(Btail[:, 1], sin.(m_otail[1]*2*dom.y); atol = 1e-4, rtol = 1e-4)

        @test isapprox(Atail[:, 2], cos.(m_etail[2]*2*dom.y); atol = 1e-4, rtol = 1e-4)
        @test isapprox(Btail[:, 2], sin.(m_otail[2]*2*dom.y); atol = 1e-4, rtol = 1e-4)

        @test isapprox(Atail[:, 5], cos.(m_etail[5]*2*dom.y); atol = 1e-4, rtol = 1e-4)
        @test isapprox(Btail[:, 5], sin.(m_otail[5]*2*dom.y); atol = 1e-4, rtol = 1e-4)
    end

    @testset "test initial condition mapping in y" begin
        Pe = 10
        Nx = 10
        Ny = 200
        M  = 1
        L  = 1/pi
        dom = domain(L, M, Nx, Ny)
        tvals = range(0, .01, 2)

        # ==================================
        # Initial condition in cross-stream direction
        # ==================================
        σ2_y0 = 1/40
        g = @. exp(-(((dom.y-pi/4)/(sqrt(2*σ2_y0))).^2))
        χ_0, χ_e, γ_o = normal_mode_decomposition(g, 2*dom.y; Lx=2π)
        γ_e = vcat(χ_0, χ_e)

        σ2_0 = 100
        f = @. 1*exp(-(((dom.x + 0)/(sqrt(2*σ2_0))).^2))
        c0, ce, co = normal_mode_decomposition(f, dom.x; Lx=2*L*π)
        Ccoeffs = vcat(c0/2, complex_x_coeffs(ce, co))

        basis = spectral_basis(:poiseuille, Pe, dom, 100; show_progress=false,compute_eigenfunctions=:true, compute_odd=:true)    
        # check initial condition in y direction
        T_serial = solve_explicit(dom, basis, γ_e, γ_o, Ccoeffs, tvals, Pe; show_progress=false,parallel=false, reconstruction=:eigenfunctions)
        T_parallel = solve_explicit(dom, basis, γ_e, γ_o, Ccoeffs, tvals, Pe; show_progress=false, reconstruction=:eigenfunctions)

        # check initial condition in y direction
        @test isapprox(T_serial[1,:,:], T_parallel[1,:,:]; atol=1e-7)
        @test isapprox(T_serial[1, :, 5], g; atol=1e-2,  rtol = 1e-4)

    end
    @testset "test initial condition mapping in x" begin
        Pe = 10
        Nx = 100
        Ny = 20
        M  = 1
        L  = 1/pi
        dom = domain(L, M, Nx, Ny)
        tvals = range(0, .01, 2)

        # ==================================
        # Initial condition in cross-stream direction
        # ==================================
        σ2_y0 = 1000
        g = @. exp(-(((dom.y-pi/4)/(sqrt(2*σ2_y0))).^2))
        χ_0, χ_e, γ_o = normal_mode_decomposition(g, 2*dom.y; Lx=2π)
        γ_e = vcat(χ_0, χ_e)

        σ2_0 = 1/100
        f = @. 1*exp(-(((dom.x + 0)/(sqrt(2*σ2_0))).^2))
        c0, ce, co = normal_mode_decomposition(f, dom.x; Lx=2*L*π)
        Ccoeffs = vcat(c0/2, complex_x_coeffs(ce, co))

        basis = spectral_basis(:poiseuille, Pe, dom, 100; show_progress=false,compute_eigenfunctions=:true, compute_odd=:true)
        T_serial = solve_explicit(dom, basis, γ_e, γ_o, Ccoeffs, tvals, Pe; show_progress=false,parallel=false, reconstruction=:eigenfunctions)
        T_parallel = solve_explicit(dom, basis, γ_e, γ_o, Ccoeffs, tvals, Pe; show_progress=false, reconstruction=:eigenfunctions)

        # check initial condition in y direction
        @test isapprox(T_serial[1,:,:], T_parallel[1,:,:]; atol=1e-7)
        @test isapprox(T_serial[1, 5, :], f; atol=1e-2,  rtol = 1e-4)

    end

end
