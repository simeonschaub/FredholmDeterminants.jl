module FredholmDeterminants

export TracyWidom

using FastGaussQuadrature, SpecialFunctions, LinearAlgebra, Distributions, ForwardDiff, DifferentiationInterface

const nodes₁, weights₁ = let (x, w) = gausslegendre(20)
    @. w *= -40 / (x^2 + 2x - 3)
    @. x = 20atanh((x + 1)/2)
    x, w
end
const α, β = 20, 0
const nodes₂, weights₂ = let (x, w) = gaussjacobi(12, α, β)
    @. w /= (1 - x)^α * (1 + x)^β
    @. w *= -20 / (x^2 + 2x - 3)
    @. x = 10atanh((x + 1)/2)
    x, w
end

K₁(x, y) = airyai((x + y) / 2) / 2
function K₂(x, y)
    if x == y
        return (airyaiprime(x))^2 - x * (airyai(x))^2
    else
        return (airyai(x) * airyaiprime(y) - airyai(y) * airyaiprime(x)) / (x - y)
    end
end

fredholm_det(K, x, w) = det(I - .√w .* K.(x, x') .* .√(w'))

struct TracyWidom{β} <: ContinuousUnivariateDistribution end

function Distributions.cdf(::TracyWidom{β}, s::Float64) where {β}
    if β == 1
        return fredholm_det((x, y) -> K₁(s + x, s + y), nodes₁, weights₁)
    elseif β == 2
        return fredholm_det((x, y) -> K₂(s + x, s + y), nodes₂, weights₂)
    else
        throw(ArgumentError("β = $β not implemented"))
    end
end

function Distributions.logpdf(::TracyWidom{β}, s::Float64) where {β}
    if β == 1
        K = K₁
        x, w = nodes₁, weights₁
    elseif β == 2
        K = K₂
        x, w = nodes₂, weights₂
    else
        throw(ArgumentError("β = $β not implemented"))
    end
    Kₛ, dKₛ = DifferentiationInterface.value_and_derivative(AutoForwardDiff(), s) do s
        .√w .* K.(s .+ x, s .+ x') .* .√(w')
    end
    A = I - Kₛ
    _logdet, sgn = logabsdet(A)
    sgn == -1 && return -Inf
    _tr = -tr(A \ dKₛ)
    _tr < 0 && return -Inf
    return _logdet + log(_tr)
    # return logdet(A) + log(-tr(A \ dKₛ), 0)
end

end
