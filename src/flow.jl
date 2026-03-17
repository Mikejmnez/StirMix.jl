

predefined_flow(flow::Symbol, dom; Nmodes=length(dom.K)-1) = predefined_flow(Val(flow), dom; Nmodes=Nmodes)

function predefined_flow(::Val{:cos}, dom; Nmodes=length(dom.K)-1)
    U = (1 .- cos.(2 .* dom.y)) ./ 2
    beta0 = 1.0
    beta = zeros(Float64, Nmodes)
    if Nmodes >= 1
        beta[1] = -0.5
    end
    return U, beta0, beta
end

function predefined_flow(::Val{:square}, dom; Nmodes=length(dom.K)-1)
    y = dom.y
    U = Float64.((π/4 .<= y) .& (y .<= 3π/4))
    beta0 = 1.0
    beta = [2 * (-1)^m / (m * pi) * sin(m * pi / 2) for m in 1:Nmodes]
    return U, beta0, beta
end