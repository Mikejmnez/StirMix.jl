
function predefined_flow(flow::Symbol, dom; Nmodes = length(dom.y) ÷ 2)
    predefined_flow(Val(flow), dom; Nmodes = Nmodes)
end

function predefined_flow(::Val{:cos}, dom; Nmodes = length(dom.y) ÷ 2)
    U = (1 .- cos.(2 .* _ytilde(dom))) ./ 2
    β0 = 1.0
    β = zeros(Float64, Nmodes)
    if Nmodes >= 1
        β[1] = -0.5
    end
    return U, β0, β, false
end

function predefined_flow(::Val{:square}, dom; Nmodes = length(dom.y) ÷ 2)
    y = _ytilde(dom)
    U = Float64.((pi / 4 .<= y) .& (y .<= 3pi / 4))
    β0 = 1.0
    β = [2 * (-1)^m / (m * pi) * sin(m * pi / 2) for m in 1:Nmodes]
    return U, β0, β, false
end

function predefined_flow(::Val{:poiseuille}, dom; Nmodes = length(dom.y) ÷ 2)
    y = _ytilde(dom)
    U = 1 .- ((y .- pi / 2) / (pi / 2)) .^ 2
    β0 = 4 / 3
    β = -[(4 / pi^2) / (m^2) for m in 1:Nmodes]
    return U, β0, β, false
end

function predefined_flow(::Val{:gaussian}, dom; Nmodes = length(dom.y) ÷ 2)
    y = _ytilde(dom)
    Ld = 0.8
    U = exp.(-(Ld^2) .* (2y .- π) .^ 2)
    β = [(1 / (Ld * sqrt(pi))) * (-1)^m * exp(-(m / (2Ld))^2) for m in 0:Nmodes]
    β0 = β[1]
    return U, β0, β[2:end], false
end

function predefined_flow(::Val{:triangular}, dom; Nmodes = length(dom.y) ÷ 2)
    y = _ytilde(dom)
    U = ifelse.(0 .<= y .< pi / 2, 2y / pi, 2 .- 2y / pi)
    β = [(2 * ((-1)^m) / ((pi^2) * (m^2))) * (1 - cos(m * pi)) for m in 1:Nmodes]
    β0 = 1
    return U, β0, β, false
end

function predefined_flow(::Val{:linear}, dom; Nmodes = length(dom.y) ÷ 2)
    y = _ytilde(dom)
    U = ifelse.(0 .<= y .< pi / 2, 2y / pi, 2 .- 2y / pi)
    β = [(2 * ((-1)^m) / ((pi^2) * (m^2))) * (1 - cos(m * pi)) for m in 1:Nmodes]
    β0 = 1
    return U, β0, β, true
end
