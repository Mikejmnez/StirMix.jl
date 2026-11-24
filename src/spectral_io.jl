const _SPECTRAL_DATA_SCHEMA_VERSION = 1
const _SPECTRAL_IO_MODES = (:compute, :read, :write, :refresh)

"""
    _normalize_spectral_io(filename, spectral_io) -> (path, mode)

Normalize the local spectral-data I/O options used by `spectral_basis`.

`spectral_io` may be a `Symbol` or string-like value and must normalize to one
of `:compute`, `:read`, `:write`, or `:refresh`. A filename is required for all
modes except `:compute`. This helper validates the option shape only; it does
not check whether a file exists or whether stored metadata matches the current
call.
"""
function _normalize_spectral_io(filename, spectral_io)
    mode = spectral_io isa Symbol ? spectral_io : Symbol(spectral_io)
    mode in _SPECTRAL_IO_MODES ||
        throw(ArgumentError("spectral_io must be one of $(_SPECTRAL_IO_MODES); got $(repr(spectral_io))"))

    if filename === nothing
        mode === :compute ||
            throw(ArgumentError("filename is required when spectral_io=$(repr(mode))"))
        return nothing, mode
    end

    filename isa AbstractString ||
        throw(ArgumentError("filename must be a local path string; got $(typeof(filename))"))

    path = String(filename)
    isempty(path) && throw(ArgumentError("filename must not be empty"))

    return path, mode
end

function _stirmix_version_string()
    v = try
        pkgversion(@__MODULE__)
    catch
        nothing
    end

    return v === nothing ? "" : string(v)
end

"""
    _spectral_io_metadata(domain, Pe, Ns, beta0, beta; kwargs...) -> NamedTuple

Build the metadata payload written alongside spectral arrays.

Writes the values for debugging and for future strict validation, but
read mode currently does not validate them with the active `spectral_basis`
arguments. Array-valued entries are copied so the metadata snapshot cannot
be mutated by later changes to `domain.K` or the Fourier coefficients.
"""
function _spectral_io_metadata(domain::Domain, Pe, Ns, β0, β;
        precision_policy,
        qc0,
        qc,
        compute_odd::Bool,
        even_extension::Bool)
    return (;
        Pe = Float64(Pe),
        Ns = Ns,
        precision_policy = String(Symbol(precision_policy)),
        qc0 = qc0,
        qc = qc,
        compute_odd = compute_odd,
        even_extension = even_extension,
        domain_L = domain.L,
        domain_M = domain.M,
        domain_Nx = domain.Nx,
        domain_Ny = domain.Ny,
        K = copy(domain.K),
        beta0 = Float64(β0),
        beta = copy(β),
        bigfloat_precision_bits_at_write = precision(BigFloat)
    )
end

"""
    _write_optional_meta!(file, key, value)

Write an optional metadata field to a JLD2 file.

Every optional entry gets a `meta/<key>_isnothing` marker. Non-`nothing` values
also get `meta/<key>`. This keeps `nothing` distinct from a missing metadata
field without relying on JLD2 to serialize `nothing`.
"""
function _write_optional_meta!(f, key::AbstractString, value)
    f["meta/$(key)_isnothing"] = value === nothing
    value === nothing && return nothing

    f["meta/$key"] = value
    return nothing
end

"""
    _write_spectral_entries!(file, name, values)

Write a ragged vector of spectral entries into `data/<name>`.

The adaptive eigensolve produces vectors and matrices whose sizes can vary with
q, so entries are stored one dataset per q-index instead of as one dense array.
Keys are zero-padded to keep the on-disk layout easy to inspect. The key width
is computed from the number of entries as `max(6, ndigits(length(values)))`, so
small files keep keys like `000001`, while files with one million or more
entries use wider keys. The chosen width is stored in `data/<name>/key_width`.
"""
function _write_spectral_entries!(f, name::AbstractString, values)
    width = _spectral_entry_key_width(length(values))
    f["data/$name/length"] = length(values)
    f["data/$name/key_width"] = width
    for k in eachindex(values)
        f[_spectral_entry_key(name, k, width)] = values[k]
    end
    return nothing
end

"""
    _spectral_entry_key_width(n) -> Int

Return the zero-padding width for `n` spectral entries.

The width is based on the number of q-indexed entries, not the q values. It is
`max(6, ndigits(n))`, so existing small-file keys stay six digits wide and very
large sweeps widen automatically.
"""
function _spectral_entry_key_width(n::Integer)
    n >= 0 || throw(ArgumentError("number of spectral entries must be nonnegative; got $n"))
    return max(6, ndigits(n))
end

"""
    _spectral_entry_key(name, k, width=6) -> String

Build the JLD2 dataset key for entry `k` in ragged entry group `name`.

`width` is normally the value from `_spectral_entry_key_width(length(values))`
and is stored as `data/<name>/key_width` when writing. The default width of 6
keeps reads compatible with files written before `key_width` existed.
"""
function _spectral_entry_key(name::AbstractString, k::Integer, width::Integer = 6)
    width > 0 ||
        throw(ArgumentError("spectral entry key width must be positive; got $width"))
    return "data/$name/$(lpad(k, width, '0'))"
end

function _require_spectral_key(f, key::AbstractString)
    haskey(f, key) ||
        throw(ArgumentError("filename is missing required dataset '$key'"))
    return f[key]
end

function _read_optional_meta(f, key::AbstractString)
    isnothing_key = "meta/$(key)_isnothing"
    Bool(_require_spectral_key(f, isnothing_key)) && return nothing
    return _require_spectral_key(f, "meta/$key")
end

function _read_spectral_key_width(f, name::AbstractString)
    width_key = "data/$name/key_width"
    return haskey(f, width_key) ? Int(f[width_key]) : 6
end

"""
    _read_spectral_entries(file, name, n) -> Vector{Any}

Read `n` ragged spectral entries from `data/<name>`.

This helper performs schema-presence checks only. It does not validate element
types, dimensions, or parameter metadata. New files store
`data/<name>/key_width`; files written before that field existed are read with
the original width of 6.
"""
function _read_spectral_entries(f, name::AbstractString, n::Integer)
    values = Vector{Any}(undef, n)
    width = _read_spectral_key_width(f, name)
    for k in 1:n
        values[k] = _require_spectral_key(f, _spectral_entry_key(name, k, width))
    end
    return values
end

"""
    open_metadata(filename) -> NamedTuple

Read the metadata and storage summary from a local spectral-data JLD2 file.

This is a convenience for inspecting or reusing data written by
`spectral_basis(...; filename, spectral_io=:write)`. It returns the stored
call context, including `Pe`, `Ns`, precision settings, domain fields, Fourier
coefficients, and storage facts such as `nk`, `has_odd`, and per-group
`key_widths`. Like `spectral_io=:read`, this function only reads what is in the
file; it does not validate the metadata against a new `spectral_basis` call.
"""
function open_metadata(filename::AbstractString)
    isfile(filename) ||
        throw(ArgumentError("filename does not exist: $filename"))

    return jldopen(filename, "r") do f
        has_odd = Bool(_require_spectral_key(f, "data/has_odd"))
        key_widths = has_odd ?
                     (;
            qs = _read_spectral_key_width(f, "qs"),
            an = _read_spectral_key_width(f, "an"),
            Ak = _read_spectral_key_width(f, "Ak"),
            bn = _read_spectral_key_width(f, "bn"),
            Bk = _read_spectral_key_width(f, "Bk")
        ) :
                     (;
            qs = _read_spectral_key_width(f, "qs"),
            an = _read_spectral_key_width(f, "an"),
            Ak = _read_spectral_key_width(f, "Ak")
        )

        return (;
            schema_version = Int(_require_spectral_key(f, "meta/schema_version")),
            stirmix_version = String(_require_spectral_key(f, "meta/stirmix_version")),
            julia_version = String(_require_spectral_key(f, "meta/julia_version")),
            Pe = Float64(_require_spectral_key(f, "meta/Pe")),
            Ns = _read_optional_meta(f, "Ns"),
            precision_policy = Symbol(_require_spectral_key(f, "meta/precision_policy")),
            qc0 = _read_optional_meta(f, "qc0"),
            qc = _read_optional_meta(f, "qc"),
            compute_odd = Bool(_require_spectral_key(f, "meta/compute_odd")),
            even_extension = Bool(_require_spectral_key(f, "meta/even_extension")),
            domain_L = Float64(_require_spectral_key(f, "meta/domain_L")),
            domain_M = Int(_require_spectral_key(f, "meta/domain_M")),
            domain_Nx = Int(_require_spectral_key(f, "meta/domain_Nx")),
            domain_Ny = Int(_require_spectral_key(f, "meta/domain_Ny")),
            K = Vector{Float64}(_require_spectral_key(f, "meta/K")),
            beta0 = Float64(_require_spectral_key(f, "meta/beta0")),
            beta = Vector{Float64}(_require_spectral_key(f, "meta/beta")),
            bigfloat_precision_bits_at_write = Int(_require_spectral_key(
                f, "meta/bigfloat_precision_bits_at_write")),
            nk = Int(_require_spectral_key(f, "data/nk")),
            has_odd = has_odd,
            key_widths = key_widths
        )
    end
end

"""
    _write_spectral_data(path, basis, meta)

Write a `SpectralBasis` to a local JLD2 file.

The file stores explicit metadata and spectral arrays rather than serializing
the `SpectralBasis` struct directly. Data is written to a temporary file in the
target directory and then moved into place, so callers either see the old file
or the complete new file. This function is meant to run after spectral
computation, never inside the threaded q-batch loop.
"""
function _write_spectral_data(path::AbstractString, basis::SpectralBasis, meta)
    dir = dirname(path)
    isempty(dir) && (dir = ".")
    isdir(dir) || mkpath(dir)

    tmp = tempname(dir)
    has_odd = basis.bn !== nothing && basis.Bk !== nothing

    try
        jldopen(tmp, "w") do f
            f["meta/schema_version"] = _SPECTRAL_DATA_SCHEMA_VERSION
            f["meta/stirmix_version"] = _stirmix_version_string()
            f["meta/julia_version"] = string(VERSION)

            f["meta/Pe"] = meta.Pe
            _write_optional_meta!(f, "Ns", meta.Ns)
            f["meta/precision_policy"] = meta.precision_policy
            _write_optional_meta!(f, "qc0", meta.qc0)
            _write_optional_meta!(f, "qc", meta.qc)
            f["meta/compute_odd"] = meta.compute_odd
            f["meta/even_extension"] = meta.even_extension
            f["meta/domain_L"] = meta.domain_L
            f["meta/domain_M"] = meta.domain_M
            f["meta/domain_Nx"] = meta.domain_Nx
            f["meta/domain_Ny"] = meta.domain_Ny
            f["meta/K"] = meta.K
            f["meta/beta0"] = meta.beta0
            f["meta/beta"] = meta.beta
            f["meta/bigfloat_precision_bits_at_write"] = meta.bigfloat_precision_bits_at_write

            f["data/nk"] = length(basis.qs)
            f["data/has_odd"] = has_odd
            _write_spectral_entries!(f, "qs", basis.qs)
            _write_spectral_entries!(f, "an", basis.an)
            _write_spectral_entries!(f, "Ak", basis.Ak)

            if has_odd
                _write_spectral_entries!(f, "bn", basis.bn)
                _write_spectral_entries!(f, "Bk", basis.Bk)
            end
        end

        mv(tmp, path; force = true)
    catch
        isfile(tmp) && rm(tmp; force = true)
        rethrow()
    end

    return nothing
end

"""
    _read_spectral_data(path) -> SpectralBasis

Read a local JLD2 spectral-data file and reconstruct a `SpectralBasis`.

Phase 1 intentionally checks only that the required datasets are present. It
does not compare stored metadata against the caller's current `Pe`, `domain`,
`Ns`, precision settings, or flow coefficients.
"""
function _read_spectral_data(path::AbstractString)
    isfile(path) ||
        throw(ArgumentError("filename does not exist: $path"))

    return jldopen(path, "r") do f
        nk = Int(_require_spectral_key(f, "data/nk"))
        has_odd = Bool(_require_spectral_key(f, "data/has_odd"))

        qs = _read_spectral_entries(f, "qs", nk)
        an = _read_spectral_entries(f, "an", nk)
        Ak = _read_spectral_entries(f, "Ak", nk)
        bn = has_odd ? _read_spectral_entries(f, "bn", nk) : nothing
        Bk = has_odd ? _read_spectral_entries(f, "Bk", nk) : nothing

        β0 = Float64(_require_spectral_key(f, "meta/beta0"))
        β = Vector{Float64}(_require_spectral_key(f, "meta/beta"))
        Pe = Float64(_require_spectral_key(f, "meta/Pe"))
        even_extension = Bool(_require_spectral_key(f, "meta/even_extension"))

        SpectralBasis(qs, β0, β, an, bn, Ak, Bk, Pe, even_extension)
    end
end
