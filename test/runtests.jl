using StirMix
using JLD2
using Test

struct ProgressTestTimes{T}
    values::Vector{T}
    delay::Float64
end

ProgressTestTimes(values; delay=0.12) = ProgressTestTimes(collect(values), delay)

struct SparseTestEntries{T} <: AbstractVector{T}
    len::Int
    entries::Dict{Int,T}
end

Base.size(entries::SparseTestEntries) = (entries.len,)
Base.IndexStyle(::Type{<:SparseTestEntries}) = IndexLinear()
Base.getindex(entries::SparseTestEntries, i::Int) = entries.entries[i]
Base.eachindex(entries::SparseTestEntries) = sort!(collect(keys(entries.entries)))

Base.length(tvals::ProgressTestTimes) = length(tvals.values)

function Base.iterate(tvals::ProgressTestTimes, state=1)
    state > length(tvals.values) && return nothing
    sleep(tvals.delay)
    return tvals.values[state], state + 1
end

"""
    identical_bases_iter(U, dom, Pe, ft, Ns; nt=1)

Build a `BasesIter`-shaped `NamedTuple` (see `StirMix.compute_bases`) whose
`basis` entries are all constructed from the *same* `Pe`, regardless of how
many distinct time-resampled levels `ft` has. Unlike `compute_bases`, which
scales `Pe` by each distinct magnitude in `ft`, this keeps every basis (and
every entry of `Pes`) identical to a plain `spectral_basis(U, Pe, dom, Ns)`.

This lets tests exercise the `BasesIter`-driven dispatch path in
`solve_explicit`/`evolve_explicit_avg` (interval lookup via `.indT`/`.ivals`,
basis/Pe selection via `.basis`/`.Pes`) while asserting the result is
identical to calling the single-basis method directly with `bases.basis[1]`.
"""
function identical_bases_iter(U, dom, Pe::Real, ft::Vector{Float64}, Ns::Integer; nt::Integer=1)
    nft, _, _, vals_pos, ivals_pos = StirMix.time_resample(ft; nt=nt)
    basis = [spectral_basis(U, Pe, dom, Ns; show_progress=false) for _ in vals_pos]
    Pes = fill(Float64(Pe), length(nft))
    indT = StirMix.indt_intervals(ivals_pos)
    return (basis=basis, nft=nft, Pes=Pes, vals=vals_pos, ivals=ivals_pos, indT=indT)
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
    dom = domain(pi, 1, 8, 50)
    fy_odd = @. sin.(2dom.y)
    fy_even = @. cos.(2dom.y)
    Pe = 1.0
    Pe_large = 1000.0

    @testset "spectral basis toggle" begin
        basis = spectral_basis(:cos, Pe, dom, 10; 
                               show_progress=false)
        @test length(basis.qs) == length(dom.K)

        mixed_basis = spectral_basis(:cos, Pe_large, dom, 10;
                                     show_progress=false,
                                     precision_policy=:mixed, 
                                     qc0=5000, qc=12000)
        @test any(q -> q isa ComplexF64, mixed_basis.qs)
        @test any(q -> q isa Complex{BigFloat}, mixed_basis.qs)

        even_basis = spectral_basis(:cos, Pe, dom, 10;
                                    show_progress=false)

        odd_coeff_basis = spectral_basis(:cos, Pe, dom, 10;
                                         show_progress=false,
                                         compute_odd=true)
        @test all(k -> odd_coeff_basis.an[k] !== nothing, eachindex(dom.K))
        @test all(k -> odd_coeff_basis.Ak[k] !== nothing, eachindex(dom.K))
        @test all(k -> odd_coeff_basis.bn[k] !== nothing, eachindex(dom.K))
        @test all(k -> odd_coeff_basis.Bk[k] !== nothing, eachindex(dom.K))

        @test all(k -> length(odd_coeff_basis.bn[k]) == size(odd_coeff_basis.Bk[k], 2),
                  eachindex(dom.K))

        float64_progress_basis = redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                spectral_basis(:cos, Pe, dom, 10;
                               precision_policy=:float64)
            end
        end
        mixed_progress_basis = redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                spectral_basis(:cos, Pe_large, dom, 10;
                               precision_policy=:mixed,
                               qc0=5000, qc=15000)
            end
        end

        @test length(float64_progress_basis.qs) == length(dom.K)
        @test length(mixed_progress_basis.qs) == length(dom.K)
        @test any(q -> q isa Complex{BigFloat}, mixed_progress_basis.qs)
    end

    @testset "spectral basis parallel chunks and progress" begin
        serial_basis, serial_output = capture_stderr() do
            spectral_basis(:cos, Pe, dom, 10;
                           precision_policy=:float64,
                           parallel=false,
                           compute_odd=true,
                           sweep_chunk_size=Int32(2))
        end
        parallel_basis, parallel_output = capture_stderr() do
            spectral_basis(:cos, Pe, dom, 10;
                           precision_policy=:float64,
                           compute_odd=true,
                           sweep_chunk_size=2)
        end

        @test serial_basis.qs == parallel_basis.qs
        @test all(k -> serial_basis.an[k] == parallel_basis.an[k], eachindex(dom.K))
        @test all(k -> serial_basis.Ak[k] == parallel_basis.Ak[k], eachindex(dom.K))
        @test all(k -> serial_basis.bn[k] == parallel_basis.bn[k], eachindex(dom.K))
        @test all(k -> serial_basis.Bk[k] == parallel_basis.Bk[k], eachindex(dom.K))
        @test isempty(serial_output) || occursin("Computing spectral basis", serial_output)
        @test isempty(parallel_output) || occursin("Computing spectral basis", parallel_output)
        @test !occursin("aborted", lowercase(serial_output))
        @test !occursin("aborted", lowercase(parallel_output))

        mixed_serial = spectral_basis(:cos, Pe_large, dom, 10;
                                      precision_policy=:mixed,
                                      qc0=5000, qc=15000,
                                      parallel=false,
                                      sweep_chunk_size=2,
                                      show_progress=false,
                                      compute_odd=true)
        mixed_parallel = spectral_basis(:cos, Pe_large, dom, 10;
                                        precision_policy=:mixed,
                                        qc0=5000, qc=15000,
                                        parallel=true,
                                        sweep_chunk_size=2,
                                        show_progress=false,
                                        compute_odd=true)

        @test mixed_serial.qs == mixed_parallel.qs
        @test any(q -> q isa Complex{BigFloat}, mixed_parallel.qs)
        @test all(k -> mixed_serial.an[k] == mixed_parallel.an[k], eachindex(dom.K))
        @test all(k -> mixed_serial.Ak[k] == mixed_parallel.Ak[k], eachindex(dom.K))
        @test all(k -> mixed_serial.bn[k] == mixed_parallel.bn[k], eachindex(dom.K))
        @test all(k -> mixed_serial.Bk[k] == mixed_parallel.Bk[k], eachindex(dom.K))

        @test_throws ArgumentError spectral_basis(:cos, Pe, dom, 10;
                                                  show_progress=:false,
                                                  precision_policy=:float64,
                                                  sweep_chunk_size=0)
    end

    @testset "spectral basis local file I/O" begin
        tmp = mktempdir()
        even_path = joinpath(tmp, "spectral-even.jld2")
        odd_path = joinpath(tmp, "spectral-odd.jld2")
        compute_path = joinpath(tmp, "spectral-compute.jld2")

        written_even = spectral_basis(:cos, Pe, dom, 10;
                                      show_progress=false,
                                      filename=even_path,
                                      spectral_io=:write)
        @test isfile(even_path)

        even_meta = StirMix.open_metadata(even_path)
        @test even_meta.schema_version == StirMix._SPECTRAL_DATA_SCHEMA_VERSION
        @test even_meta.Pe == Float64(Pe)
        @test even_meta.Ns == 10
        @test even_meta.precision_policy === :float64
        @test even_meta.qc0 == 1000.0
        @test even_meta.qc === nothing
        @test even_meta.compute_odd === false
        @test even_meta.even_extension === false
        @test even_meta.nk == length(dom.K)
        @test even_meta.has_odd === false
        @test even_meta.key_widths == (qs=6, an=6, Ak=6)
        @test even_meta.domain_L == dom.L
        @test even_meta.domain_M == dom.M
        @test even_meta.domain_Nx == dom.Nx
        @test even_meta.domain_Ny == dom.Ny
        @test even_meta.K == dom.K
        @test even_meta.beta0 == written_even.β0
        @test even_meta.beta == written_even.β
        @test even_meta.bigfloat_precision_bits_at_write == precision(BigFloat)

        metadata_domain = domain(even_meta.domain_L, even_meta.domain_M,
                                 even_meta.domain_Nx, even_meta.domain_Ny)
        @test metadata_domain.K == even_meta.K

        metadata_basis = spectral_basis(:cos, even_meta.Pe, metadata_domain, even_meta.Ns;
                                        show_progress=false,
                                        filename=even_path,
                                        spectral_io=:read)
        @test metadata_basis.qs == written_even.qs
        @test metadata_basis.an == written_even.an
        @test metadata_basis.Ak == written_even.Ak

        read_even = spectral_basis(:cos, Pe_large, dom, 10;
                                   show_progress=false,
                                   filename=even_path,
                                   spectral_io=:read)
        @test read_even.qs == written_even.qs
        @test read_even.β0 == written_even.β0
        @test read_even.β == written_even.β
        @test read_even.an == written_even.an
        @test read_even.Ak == written_even.Ak
        @test read_even.bn === nothing
        @test read_even.Bk === nothing
        @test read_even.Pe == written_even.Pe
        @test read_even.even_extension == written_even.even_extension

        compute_only = spectral_basis(:cos, Pe, dom, 10;
                                      show_progress=false,
                                      filename=compute_path,
                                      spectral_io=:compute)
        @test length(compute_only.qs) == length(dom.K)
        @test !isfile(compute_path)

        written_odd = spectral_basis(:cos, Pe, dom, 10;
                                     show_progress=false,
                                     compute_odd=true,
                                     filename=odd_path,
                                     spectral_io=:refresh)
        @test isfile(odd_path)
        odd_meta = StirMix.open_metadata(odd_path)
        @test odd_meta.compute_odd === true
        @test odd_meta.has_odd === true
        @test odd_meta.key_widths == (qs=6, an=6, Ak=6, bn=6, Bk=6)

        read_odd = spectral_basis(:cos, Pe_large, dom, 10;
                                  show_progress=false,
                                  filename=odd_path,
                                  spectral_io="read")
        @test read_odd.qs == written_odd.qs
        @test read_odd.an == written_odd.an
        @test read_odd.Ak == written_odd.Ak
        @test read_odd.bn == written_odd.bn
        @test read_odd.Bk == written_odd.Bk

        # test that errs when file not there.
        @test_throws ArgumentError spectral_basis(:cos, Pe, dom, 10;
                                                  show_progress=false,
                                                  filename=joinpath(tmp, "missing.jld2"),
                                                  spectral_io=:read)
        @test_throws ArgumentError StirMix.open_metadata(joinpath(tmp, "missing-metadata.jld2"))

        # test that filename is required
        @test_throws ArgumentError spectral_basis(:cos, Pe, dom, 10;
                                                  show_progress=false,
                                                  spectral_io=:write)
    end

    @testset "spectral I/O helper behavior" begin
        tmp = mktempdir()

        path, mode = StirMix._normalize_spectral_io(nothing, :compute)
        @test path === nothing
        @test mode === :compute

        path, mode = StirMix._normalize_spectral_io(SubString("basis.jld2", 1, 10), "read")
        @test path == "basis.jld2"
        @test mode === :read

        @test_throws ArgumentError StirMix._normalize_spectral_io(nothing, :read)
        @test_throws ArgumentError StirMix._normalize_spectral_io("", :read)
        @test_throws ArgumentError StirMix._normalize_spectral_io(10, :read)
        # the behavior below is not yet supported
        @test_throws ArgumentError StirMix._normalize_spectral_io("basis.jld2", :reuse)

        _, β0, β, even_extension = StirMix.flow_decomposition(:cos, dom)
        meta = StirMix._spectral_io_metadata(dom, Pe, nothing, β0, β;
                                             precision_policy=:mixed,
                                             qc0=5000,
                                             qc=15000,
                                             compute_odd=true,
                                             even_extension=even_extension)
        @test meta.Pe == Float64(Pe)
        @test meta.Ns === nothing
        @test meta.precision_policy == "mixed"
        @test meta.qc0 == 5000
        @test meta.qc == 15000
        @test meta.compute_odd === true
        @test meta.even_extension == even_extension
        @test meta.K == dom.K
        @test meta.K !== dom.K
        @test meta.beta == β
        @test meta.beta !== β
        @test meta.bigfloat_precision_bits_at_write == precision(BigFloat)

        optional_path = joinpath(tmp, "optional-meta.jld2")
        jldopen(optional_path, "w") do f
            StirMix._write_optional_meta!(f, "qc", nothing)
            StirMix._write_optional_meta!(f, "Ns", 10)
        end
        jldopen(optional_path, "r") do f
            @test f["meta/qc_isnothing"] === true
            @test !haskey(f, "meta/qc")
            @test f["meta/Ns_isnothing"] === false
            @test f["meta/Ns"] == 10
        end

        entries_path = joinpath(tmp, "entries.jld2")
        values = Any[ComplexF64[1], ComplexF64[2, 3]]
        jldopen(entries_path, "w") do f
            StirMix._write_spectral_entries!(f, "vals", values)
        end
        jldopen(entries_path, "r") do f
            @test StirMix._spectral_entry_key("vals", 2) == "data/vals/000002"
            @test StirMix._spectral_entry_key_width(2) == 6
            @test f["data/vals/length"] == 2
            @test f["data/vals/key_width"] == 6
            @test StirMix._read_spectral_entries(f, "vals", 2) == values
            @test_throws ArgumentError StirMix._read_spectral_entries(f, "vals", 3)
        end
        # this test if for when there are a lot of q-entries being stored
        # necessary since writing q-entries pads with 6 zeros the keys by default
        wide_entries_path = joinpath(tmp, "wide-entries.jld2")
        wide_values = SparseTestEntries(1_000_000, Dict(1 => "first", 1_000_000 => "last"))
        jldopen(wide_entries_path, "w") do f
            StirMix._write_spectral_entries!(f, "wide", wide_values)
        end
        jldopen(wide_entries_path, "r") do f
            @test StirMix._spectral_entry_key_width(999_999) == 6
            @test StirMix._spectral_entry_key_width(1_000_000) == 7
            @test StirMix._spectral_entry_key("wide", 1, 7) == "data/wide/0000001"
            @test f["data/wide/length"] == 1_000_000
            @test f["data/wide/key_width"] == 7
            @test f["data/wide/0000001"] == "first"
            @test f["data/wide/1000000"] == "last"
            @test !haskey(f, "data/wide/000001")
        end

        legacy_entries_path = joinpath(tmp, "legacy-entries.jld2")
        jldopen(legacy_entries_path, "w") do f
            f["data/legacy/length"] = 2
            f["data/legacy/000001"] = "old-first"
            f["data/legacy/000002"] = "old-second"
        end
        jldopen(legacy_entries_path, "r") do f
            @test !haskey(f, "data/legacy/key_width")
            @test StirMix._read_spectral_entries(f, "legacy", 2) == Any["old-first", "old-second"]
        end

        basis = StirMix.SpectralBasis(
            Any[0.0 + 0.0im, 1.0im],
            1.0,
            [-0.5],
            Any[ComplexF64[1], ComplexF64[2, 3]],
            Any[ComplexF64[4], ComplexF64[5, 6]],
            Any[reshape(ComplexF64[1], 1, 1), ComplexF64[1 0; 0 1]],
            Any[reshape(ComplexF64[1], 1, 1), ComplexF64[0 1; 1 0]],
            Pe,
            false,
        )

        spectral_path = joinpath(tmp, "nested", "basis.jld2")
        StirMix._write_spectral_data(spectral_path, basis, meta)
        @test isfile(spectral_path)

        jldopen(spectral_path, "r") do f
            @test f["meta/schema_version"] == StirMix._SPECTRAL_DATA_SCHEMA_VERSION
            @test f["meta/Ns_isnothing"] === true
            @test f["meta/precision_policy"] == "mixed"
            @test f["data/nk"] == 2
            @test f["data/has_odd"] === true
            @test f["data/an/length"] == 2
            @test haskey(f, "data/Bk/000002")
        end

        loaded = StirMix._read_spectral_data(spectral_path)
        @test loaded.qs == basis.qs
        @test loaded.β0 == basis.β0
        # β is reconstructed from metadata, not from a raw serialized basis field.
        @test loaded.β == meta.beta
        @test loaded.an == basis.an
        @test loaded.bn == basis.bn
        @test loaded.Ak == basis.Ak
        @test loaded.Bk == basis.Bk
        @test loaded.Pe == basis.Pe
        @test loaded.even_extension == basis.even_extension

        malformed_path = joinpath(tmp, "malformed.jld2")
        jldopen(malformed_path, "w") do f
            f["data/nk"] = 1
            f["data/has_odd"] = false
            f["data/qs/000001"] = 0.0 + 0.0im
            f["meta/beta0"] = 1.0
            f["meta/beta"] = [-0.5]
            f["meta/Pe"] = Pe
            f["meta/even_extension"] = false
        end
        @test_throws ArgumentError StirMix._read_spectral_data(malformed_path)
    end

    @testset "averaged preparation reuses mixed spectral data" begin
        ccoeffs = ones(ComplexF64, length(dom.K))

        basis = spectral_basis(:cos, Pe, dom, 10; show_progress=false)
        field = solve_explicit(dom, basis, fy_even, ccoeffs, [0.1];
                               show_progress=false)
        @test size(field) == (1, length(dom.y), length(dom.x))

        basis = spectral_basis(:cos, Pe_large, dom, 10; precision_policy=:mixed,
                                     qc0=5000, qc = 15000,
                                     compute_odd=false,
                                     show_progress=false)
        avg_field = evolve_explicit_avg(dom, basis, fy_even, ccoeffs, [0.1]; 
                                        show_progress=false)

        @test size(avg_field) == (1, length(dom.x))
    end

    @testset "explicit evolution progress" begin
        ccoeffs = ones(ComplexF64, length(dom.K))
        tvals = [0.1, 0.2, 0.3]
        progress_tvals = ProgressTestTimes(tvals)

        basis = spectral_basis(:cos, Pe, dom, 10; precision_policy=:float64, show_progress=false)
        field = solve_explicit(dom, basis, fy_even, ccoeffs, tvals; show_progress=false)
        parallel_field = solve_explicit(dom, basis, fy_even, ccoeffs, tvals;
                                        parallel=true, show_progress=false)
        coefficient_field = solve_explicit(dom, basis, fy_even, ccoeffs, tvals;
                                           show_progress=false)

        coefficient_basis = spectral_basis(:cos, Pe, dom, 10;
                                           show_progress=false,
                                           precision_policy=:float64,
                                           compute_odd=false)
        coefficient_only_field = solve_explicit(dom, coefficient_basis, fy_even,
                                                ccoeffs, tvals;
                                                show_progress=false)

        no_progress_field, no_progress_output = capture_stderr() do
            solve_explicit(dom, basis, fy_even, ccoeffs, progress_tvals;
                           show_progress=false)
        end
        progress_field, progress_output = capture_stderr() do
            solve_explicit(dom, basis, fy_even, ccoeffs, progress_tvals;
                           show_progress=true)
        end

        avg_field = evolve_explicit_avg(dom, basis, fy_even, ccoeffs, tvals, show_progress=false)
        parallel_avg = evolve_explicit_avg(dom, basis, fy_even, ccoeffs, tvals;
                                           parallel=true, show_progress=false)

        no_progress_avg, no_progress_avg_output = capture_stderr() do
            evolve_explicit_avg(dom, basis, fy_even, ccoeffs, progress_tvals;
                                show_progress=false)
        end
        progress_avg, progress_avg_output = capture_stderr() do
            evolve_explicit_avg(dom, basis, fy_even, ccoeffs, progress_tvals;
                                show_progress=true)
        end

        @test no_progress_field == field
        @test progress_field == field
        @test parallel_field ≈ field
        @test coefficient_field ≈ field
        @test coefficient_only_field ≈ field
        @test coefficient_basis.bn === nothing
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

    @testset "solve_explicit tvals type compatibility" begin
        ccoeffs = ones(ComplexF64, length(dom.K))
        basis = spectral_basis(:cos, Pe, dom, 10; precision_policy=:float64, show_progress=false)

        reference = solve_explicit(dom, basis, fy_even, ccoeffs, [0.1, 0.2, 0.3]; show_progress=false)

        for tvals in [0.1:0.1:0.3, ProgressTestTimes([0.1, 0.2, 0.3])]
            @test solve_explicit(dom, basis, fy_even, ccoeffs, tvals; show_progress=false) ≈ reference
        end

        scalar_ref = solve_explicit(dom, basis, fy_even, ccoeffs, [0.1]; show_progress=false)
        @test solve_explicit(dom, basis, fy_even, ccoeffs, 0.1; show_progress=false) ≈ scalar_ref
    end

    @testset "time resampling ranges" begin
        nft, vals, ivals = StirMix.time_resample([0.0, 0.24, 0.26, 0.74, 0.76, 1.0];
                                                 nt=1)
        @test vals == [0.0, 0.5, 1.0]
        @test nft == [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]
        @test ivals == [1, 1, 2, 2, 3, 3]
        @test vals[ivals] == nft

        signed_nft, signed_vals, signed_ivals =
            StirMix.time_resample([-1.0, -0.74, -0.26, 0.26, 0.74, 1.0]; nt=1)
        @test signed_vals == [-1.0, -0.5, 0.0, 0.5, 1.0]
        @test signed_nft == [-1.0, -0.5, -0.5, 0.5, 0.5, 1.0]
        @test signed_ivals == [1, 2, 2, 4, 4, 5]
        @test signed_vals[signed_ivals] == signed_nft
    end

    @testset "explicit evolution" begin
        ccoeffs = ones(ComplexF64, length(dom.K))
        tvals = [0.1]
        Pe=10.0

        basis_coeff = spectral_basis(:cos, Pe, dom, 4;
                                     precision_policy=:mixed,
                                     show_progress=false,
                                     qc0=0.0,
                                     qc=1.0e3,
                                     compute_odd=false)
        basis_odd_coeff = spectral_basis(:cos, Pe, dom, 4;
                                         precision_policy=:mixed,
                                         show_progress=false,
                                         qc0=0.0,
                                         qc=1.0e3,
                                         compute_odd=true)

        field_coeff = solve_explicit(dom, basis_coeff, fy_even, ccoeffs,
                                     tvals; show_progress=false)
        field_odd_coeff = solve_explicit(dom, basis_odd_coeff, fy_odd,
                                         ccoeffs, tvals;
                                         show_progress=false)

        @test any(k -> eltype(basis_coeff.Ak[k]) === Complex{BigFloat},
                  eachindex(dom.K))
        @test any(k -> eltype(basis_odd_coeff.Bk[k]) === Complex{BigFloat},
                  eachindex(dom.K))
        @test_throws ArgumentError solve_explicit(dom, basis_coeff, fy_odd,
                                                  ccoeffs, tvals;
                                                  show_progress=false)


        @test size(field_coeff) == size(field_coeff)
    end
    @testset "flow decomposition comparison: predetermined vs prescribed" begin
        Nx=8
        Ny=100
        dom = domain(pi, 1, Nx, Ny)
        U, β0_1, β_1, _ = StirMix.predefined_flow(:poiseuille, dom)
        β0_2, β_2, _ = StirMix.normal_mode_decomposition(U, dom.y; Lx=Ny)

        @test isapprox(real(β0_1), real(β0_2); atol=1e-2, rtol=1e-2)
        @test isapprox(real(β_1), real(β_2); atol=1e-2, rtol=1e-2)
    end

    @testset "2D normal mode decomposition along a dimension" begin
        Nx = 100
        Ny = 150
        L = 10
        M=1
        dom = domain(L, M, Nx, Ny)
        g = @. dom.y
        f = @. cos(2π * dom.x) - sin(3π * dom.x)
        Z = (f * g')'

        c0_2D, ce_2D, co_2D = StirMix.normal_mode_decomposition(Z, dom.x; Lx=L, dim=2)
        c0, ce, co = StirMix.normal_mode_decomposition(f, dom.x; Lx=L)

        diff = abs.(c0.* g - c0_2D)
        @test maximum(diff) < 1e-10

        for j in 1:length(ce)
            diff = abs.(ce[j].* g - ce_2D[:, j])
            @test maximum(diff) < 1e-10
        end

        for j in 1:length(co)
            diff = abs.(co[j].* g - co_2D[:, j])
            @test maximum(diff) < 1e-10
        end

        Ces = hcat(c0_2D, StirMix.complex_x_coeffs(ce_2D, co_2D))
        C = vcat(c0, StirMix.complex_x_coeffs(ce, co))
        χ0, χe, χo = StirMix.normal_mode_decomposition(Ces, 2 .* dom.y; Lx=2π, dim=1)
        γ0, γe, γo = StirMix.normal_mode_decomposition(g, 2 .* dom.y; Lx=2π)

        @test eltype(χ0) <: Complex
        @test χ0 ≈ γ0 .* C
        @test χe ≈ γe * transpose(C)
        @test χo ≈ γo * transpose(C)
    end

    @testset "odd flow decomposition Even periodic extension" begin
        Nx = 8
        Ny = 200
        dom = domain(pi, 1, Nx, Ny)
        # define purely odd flow field
        U_linear = dom.y / π

        U, α0, α, _ = StirMix.flow_decomposition(U_linear,  dom)

        @test round(α0/2, digits=1) === 0.5
        @test maximum(round.(abs.(α[2:2:end]), digits=10)) === 0.0
        @test minimum(round.(abs.(α[2:2:end]), digits=10)) === 0.0

    end

    @testset "tail normal modes" begin 
        Nx = 20
        Ny = 200
        M  = 1
        L  = 1
        dom = domain(L, M, Nx, Ny)

        # ==================================
        # Initial condition in cross-stream direction
        # ==================================
        σ2_y0 = 1/40
        g = @. exp(-(((dom.y-pi/4)/(sqrt(2*σ2_y0))).^2))
        χ_0, χ_e, γ_o = StirMix.normal_mode_decomposition(g, 2*dom.y; Lx=2π)
        γ_e = vcat(χ_0, χ_e)
        
        NMax = 10 # Tail data begins with the following normal mode
        Nnm = length(γ_e) - NMax # how many tail cosine modes there will be
        NNmo = length(γ_o) - NMax

        m_etail, Atail, γ_etail, _ = StirMix.cosine_tail_data(dom.y, γ_e, NMax)
        m_otail, Btail, γ_otail, _ = StirMix.sine_tail_data(dom.y, γ_o, NMax)

        @test size(Atail) == (length(dom.y), Nnm)
        @test γ_etail[1] === γ_e[NMax+1]
        @test γ_otail[1] === γ_o[NMax+1]

        @test isapprox(Atail[:, 1], cos.(NMax*2*dom.y); atol = 1e-4, rtol = 1e-4)
        @test isapprox(Btail[:, 1], sin.((NMax+1)*2*dom.y); atol = 1e-4, rtol = 1e-4)

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
        L  = 1
        dom = domain(L, M, Nx, Ny)
        tvals = range(0, .01, 2)
        newβ0 = 0
        # ==================================
        # Initial condition in cross-stream direction
        # ==================================
        σ2_y0 = 1/40
        g = @. exp(-(((dom.y-pi/4)/(sqrt(2*σ2_y0))).^2))

        σ2_0 = 100
        f = @. 1*exp(-(((dom.x + 0)/(sqrt(2*σ2_0))).^2))
        c0, ce, co = StirMix.normal_mode_decomposition(f, dom.x; Lx=2*L*π)
        Ccoeffs = vcat(c0/2, StirMix.complex_x_coeffs(ce, co))

        basis = spectral_basis(:poiseuille, Pe, dom, 100; show_progress=false, compute_odd=true)

        # check initial condition in y direction
        T_serial = solve_explicit(dom, basis, g, Ccoeffs, tvals;
                                  show_progress=false, parallel=false)
        T_parallel = solve_explicit(dom, basis, g, Ccoeffs, tvals;
                                  show_progress=false)
        T_zero_ave_flow = solve_explicit(dom, basis, g, Ccoeffs, tvals;
                                    β0=newβ0, # <---- can manually specify cross-stream flow average
                                  show_progress=false)

        # check initial condition in y direction
        @test isapprox(T_serial[1,:,:], T_parallel[1,:,:]; atol=1e-7)
        @test isapprox(T_serial[1,:,:], T_zero_ave_flow[1,:,:]; atol=1e-7)
        @test isapprox(T_serial[1, :, 5], g; atol=1e-2,  rtol = 1e-4)

    end
    @testset "test initial condition mapping in x" begin
        Pe = 10
        Nx = 100
        Ny = 20
        M  = 1
        L  = 1
        dom = domain(L, M, Nx, Ny)
        tvals = range(0, .01, 2)

        newβ0 = 0
        # ==================================
        # Initial condition in cross-stream direction
        # ==================================
        σ2_y0 = 1000
        g = @. exp(-(((dom.y-pi/4)/(sqrt(2*σ2_y0))).^2))

        σ2_0 = 1/100
        f = @. 1*exp(-(((dom.x + 0)/(sqrt(2*σ2_0))).^2))
        c0, ce, co = StirMix.normal_mode_decomposition(f, dom.x; Lx=L)
        Ccoeffs = vcat(c0/2, StirMix.complex_x_coeffs(ce, co))

        basis = spectral_basis(:poiseuille, Pe, dom, 100; show_progress=false,
                              compute_odd=true)
        T_serial = solve_explicit(dom, basis, g, Ccoeffs, tvals;
                                  show_progress=false, parallel=false)
        T_parallel = solve_explicit(dom, basis, g, Ccoeffs, tvals;
                                    show_progress=false)
        T_zero_ave_flow = solve_explicit(dom, basis, g, Ccoeffs, tvals;
                                    β0=newβ0, # <---- can manually specify cross-stream flow average
                                    show_progress=false)

                                    
        # check initial condition in y direction
        @test isapprox(T_serial[1,:,:], T_parallel[1,:,:]; atol=1e-7)
        @test isapprox(T_serial[1,:,:], T_zero_ave_flow[1,:,:]; atol=1e-7)

        @test isapprox(T_serial[1, 5, :], f; atol=1e-2,  rtol = 1e-4)

    end

    @testset "time amplitude discretization" begin
        t = range(0, 2π, length=100)
        ft = 0.5 .+ 0.5*cos.(2 .* t)
        ft2 = cos.(2 .* t)

        @testset "ft = $fname" for (fname, f) in [("nonnegative", ft), ("with negatives", ft2)]
            nft, vals, ivals = StirMix.time_resample(f; nt=1)

            @assert all(i -> nft[i] == vals[ivals[i]], eachindex(ivals))
            ind = StirMix.indt_intervals(ivals)
            diff = zeros(length(ind))
            # test that values are indeed equal in the intervals
            @assert all(i -> abs(nft[ind[i][1]] - nft[ind[i][2]]) === diff[i], range(1,length(ind)))
        end
    end


    @testset "y function 1D projection: Case1 " begin
        dom = domain(1, 1, 8, 501)
        Ld1 = 1.0 / 8.0
        @testset "Case 1: phi_new=π/2 phi_old=π" begin
            phi_new = 0.5*π
            phi_old = π
            Phi     = exp.(-0.5 .* (2dom.y .- phi_old).^2 ./ (0.5 * Ld1^2))
            expected = exp.(-0.5 .* (2dom.y .- (phi_old - phi_new)).^2 ./ (0.5 * Ld1^2))
            even_c, odd_c, nphi_new, nphi_old = StirMix.coeff_project(Phi, dom.y; phi_new = phi_new, phi_old = phi_old)
            reconstructed = StirMix.reconstruct_fourier_series(collect(2*dom.y), even_c[1], even_c[2:end], odd_c; Lx=2*π)
            @test isapprox(reconstructed, expected; atol=1e-4, rtol=1e-4)
            @test isapprox(nphi_new, phi_old - phi_new; atol=1e-4, rtol=1e-4)
            @test isapprox(nphi_old, phi_old + phi_new; atol=1e-4, rtol=1e-4)
        end
        @testset "Case 2: phi_new=π/4 phi_old=π/2" begin
            phi_new = 0.25*π
            phi_old = 0.5*π
            Phi     = exp.(-0.5 .* (2dom.y .- phi_old).^2 ./ (0.5 * Ld1^2))
            expected = exp.(-0.5 .* (2dom.y .- (phi_old - phi_new)).^2 ./ (0.5 * Ld1^2))
            even_c, odd_c, nphi_new, nphi_old = StirMix.coeff_project(Phi, dom.y; phi_new = phi_new, phi_old = phi_old)
            reconstructed = StirMix.reconstruct_fourier_series(collect(2*dom.y), even_c[1], even_c[2:end], odd_c; Lx=2*π)
            @test isapprox(reconstructed, expected; atol=1e-4, rtol=1e-4)
            @test isapprox(nphi_new, phi_old - phi_new; atol=1e-4, rtol=1e-4)
            @test isapprox(nphi_old, phi_old + phi_new; atol=1e-4, rtol=1e-4)
        end

        @testset "Case 3: phi_new=π phi_old=π" begin
            phi_new = π
            phi_old = π
            Phi     = exp.(-0.5 .* (2dom.y .- phi_old).^2 ./ (0.5 * Ld1^2))
            expected = exp.(-0.5 .* (2dom.y .- (phi_old - phi_new)).^2 ./ (0.5 * Ld1^2)) + exp.(-0.5 .* (2dom.y .- 2π).^2 ./ (0.5 * Ld1^2))

            even_c, odd_c, nphi_new, nphi_old = StirMix.coeff_project(Phi, dom.y; phi_new = phi_new)
            reconstructed = StirMix.reconstruct_fourier_series(collect(2*dom.y), even_c[1], even_c[2:end], odd_c; Lx=2*π)
            @test isapprox(reconstructed, expected; atol=1e-4, rtol=1e-4)
            @test isapprox(nphi_new, π; atol=1e-4, rtol=1e-4)
            @test isapprox(nphi_old, 2*π; atol=1e-4, rtol=1e-4)
        end
    end
    @testset "2D IVP" begin
        Pe = 1
        Nx = 1000
        Ny = 150
        M  = 1
        L  = 100
        dom = domain(L, M, Nx, Ny)
        tvals = [0.0, 0.1]
        basis = spectral_basis(:cos, Pe, dom, 100; compute_odd=true, show_progress=false)
        @testset "Case 1: separable gaussian stripe IVP" begin
            # Initial condition in cross-stream direction
            σ2_y0 = 1000
            g = @. exp(-(((dom.y-pi/2)/(sqrt(2*σ2_y0))).^2))
            σ2_x = 10
            X0 =0 
            f = @. exp(-((dom.x-X0)/sqrt(2*σ2_x)).^2)
            # get 2D initial condition from separable components
            Z = g * transpose(f)

            T = solve_explicit(dom, basis, Z, tvals; β0=0)
            @test size(T) == (length(tvals), length(dom.y), length(dom.x))
            @test isapprox(T[1, :, :], Z; atol=1e-3, rtol=1e-3)
        end
        @testset "Case 2: non-separable IVP" begin
            σ2_y0 = 1/5
            g = @. exp(-(((dom.y-pi/2)/(sqrt(2*σ2_y0))).^2))
            σ2_x = 50
            X0 =0 
            f = @. exp(-((dom.x-X0)/sqrt(2*σ2_x)).^2)
            # get 2D initial condition from separable components
            Z = g * transpose(f)
            f1 = @. cos(pi* dom.x /(L/2)) - sin(pi* dom.x /(L/4))
            f2 = @. sin(pi* dom.x /(L/2)) + cos(pi* dom.x /(L/4))
            g1 = @. cos(2pi* dom.y)
            g2 = @. sin(2pi* dom.y)
            Z1 = g1 * f1'
            Z2 = g2 * f2'
            Zn = Z .* (Z1 + Z2)
            T = solve_explicit(dom, basis, Zn, tvals; β0=0)
            @test size(T) == (length(tvals), length(dom.y), length(dom.x))
            @test isapprox(T[1, :, :], Zn; atol=1e-4, rtol=1e-1)
        end
    end

    @testset "2D IVP with time dependence" begin
        Pe = 100
        Nx = 1000
        Ny = 150
        M  = 1
        L  = 100 # how much bigger is the x-domain than cross-stream
        dom = domain(L, M, Nx, Ny)
        σ2_y0 = 1/50
        g = @. exp(-(((dom.y-pi/2)/(sqrt(2*σ2_y0))).^2))        
        σ2_x = 10
        X0 = -20
        f = @. exp(-((dom.x-X0)/sqrt(2*σ2_x)).^2)
        Z = g * transpose(f)
        
        tvals = range(0, 0.25, length=41)
        ft = sin.(10π .* tvals)

        bases = identical_bases_iter(:cos, dom, Pe, ft, 100; nt=1)
        bases_lin = identical_bases_iter(:linear, dom, Pe, ft, 100; nt=1)

        T_iter = solve_explicit(dom, bases, Z, tvals; show_progress=false)
        T_cont = solve_explicit(dom, bases.basis[1], Z, tvals; show_progress=false)
        diff = abs.(T_iter[end, :, :] - T_cont[end, :, :])

        @test isapprox(diff, zero(diff); atol=1e-4, rtol=1e-1)

        T_iter = solve_explicit(dom, bases_lin, Z, tvals; show_progress=false)
        T_cont = solve_explicit(dom, bases_lin.basis[end], Z, tvals; show_progress=false)
        diff = abs.(T_iter[end, :, :] - T_cont[end, :, :])
        @test isapprox(diff, zero(diff); atol=1e-3, rtol=1e-1)

    end

    @testset "2D IVP with time dependence but soln is y-averaged" begin
        Pe = 100
        dom = domain(200, 1, 2500, 250)

        σ2_y0 = 1/10
        g_c = @. exp(-(((dom.y-pi/2)/(sqrt(2*σ2_y0))).^2))

        tvals = range(0, 0.2, length=10)
        ft = sin.(5π .* tvals)

        # BasesIter built from identical (same-Pe) bases: exercises the
        # interval-dispatch path in evolve_explicit_avg while the physics
        # itself never actually changes basis, so it must match the
        # single-basis call.
        bases = identical_bases_iter(:cos, dom, Pe, ft, 100; nt=0)

        # single mode in x
        ik=2
        Cmodes = zero(dom.K)
        Cmodes[ik] = 1

        T_c = evolve_explicit_avg(dom, bases.basis[1], g_c, Cmodes, tvals;show_progress=false)
        T_iter = evolve_explicit_avg(dom, bases, g_c, Cmodes, tvals; show_progress=false)

        ik=4
        Cmodes = zero(dom.K)
        Cmodes[ik] = 1
        T_c = evolve_explicit_avg(dom, bases.basis[1], g_c, Cmodes, tvals; show_progress=false)
        T_iter = evolve_explicit_avg(dom, bases, g_c, Cmodes, tvals; show_progress=false)

        diff = T_c .- T_iter
        @test isapprox(diff, zero(diff); atol=1e-10, rtol=1e-1)
    end

end
