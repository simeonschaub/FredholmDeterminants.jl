module FredholmDeterminants

export TracyWidom

using FastGaussQuadrature, SpecialFunctions, LinearAlgebra, Distributions,
    ForwardDiff, DifferentiationInterface, LogExpFunctions

const nodes₁, weights₁ = let (x, w) = gausslegendre(20)
    @. w *= -40 / (x^2 + 2x - 3)
    @. x = 20atanh((x + 1) / 2)
    x, w
end
const α, β = 20, 0
const nodes₂, weights₂ = let (x, w) = gaussjacobi(12, α, β)
    @. w /= (1 - x)^α * (1 + x)^β
    @. w *= -20 / (x^2 + 2x - 3)
    @. x = 10atanh((x + 1) / 2)
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

fredholm_det(K, x, w) = det(I - Symmetric(.√w .* K.(x, x') .* .√(w')))

struct TracyWidom{β} <: ContinuousUnivariateDistribution end

function Distributions.cdf(::TracyWidom{β}, s::Real) where {β}
    if β == 1
        s < -20 && return zero(s)
        return fredholm_det((x, y) -> K₁(s + x, s + y), nodes₁, weights₁)
    elseif β == 2
        return fredholm_det((x, y) -> K₂(s + x, s + y), nodes₂, weights₂)
    elseif β == 4
        F₁, F₂ = cdf(TracyWidom{1}(), √2 * s), cdf(TracyWidom{2}(), √2 * s)
        return (F₁ + F₂ / F₁) / 2
    else
        throw(ArgumentError("β = $β not implemented"))
    end
end

function Kₛ_pushforward(K, s, x, w)
    Kₛ, dKₛ = DifferentiationInterface.value_and_derivative(AutoForwardDiff(), s) do s
        .√w .* K.(s .+ x, s .+ x') .* .√(w')
    end
    return Symmetric(Kₛ), Symmetric(dKₛ)
end

function _logpdf(::TracyWidom{β}, s::Real) where {β}
    if β == 1
        s < -9.4 && return oftype(s, -Inf)
        K = K₁
        x, w = nodes₁, weights₁
    elseif β == 2
        K = K₂
        x, w = nodes₂, weights₂
    elseif β == 4
        s < -6.05 && return oftype(s, -Inf)

        Kₛ₁, dKₛ₁ = Kₛ_pushforward(K₁, √2 * s, nodes₁, weights₁)
        A₁ = I - Kₛ₁
        _logdet₁ = logdet(A₁)
        _tr₁ = -√2 * tr(A₁ \ dKₛ₁)

        Kₛ₂, dKₛ₂ = Kₛ_pushforward(K₂, √2 * s, nodes₂, weights₂)
        A₂ = I - Kₛ₂
        _logdet₂ = logdet(A₂)
        _tr₂ = -√2 * tr(A₂ \ dKₛ₂)

        return logsubexp(
            _logdet₂ + log(_tr₂) - _logdet₁,
            _logdet₁ + log(_tr₁) + logexpm1(_logdet₂ - 2_logdet₁),
        ) - log(2)
    else
        throw(ArgumentError("β = $β not implemented"))
    end
    Kₛ, dKₛ = Kₛ_pushforward(K, s, x, w)
    A = I - Kₛ
    return logdet(A) + log(-tr(A \ dKₛ))
end

logpdf⁻(s, β) = -β / 24 * abs(s)^3
const s⁻ = (-Inf, -7.0, nothing, -Inf)
const Z⁻ = map(enumerate(s⁻)) do (β, s⁻)
    (s⁻ === nothing || !isfinite(s⁻)) && return s⁻
    logpdf⁻(s⁻, β) - _logpdf(TracyWidom{β}(), s⁻)
end

logpdf⁺(s, β) = -2 / 3 * β * abs(s)^(3 / 2)
const s⁺ = (100.0, 60.0, nothing, Inf)
const Z⁺ = map(enumerate(s⁺)) do (β, s⁺)
    (s⁺ === nothing || !isfinite(s⁺)) && return s⁺
    logpdf⁺(s⁺, β) - _logpdf(TracyWidom{β}(), s⁺)
end

function Distributions.logpdf(::TracyWidom{β}, s::Real) where {β}
    if s < s⁻[β]
        return logpdf⁻(s, β) - Z⁻[β]
    elseif s > s⁺[β]
        return logpdf⁺(s, β) - Z⁺[β]
    end
    return _logpdf(TracyWidom{β}(), s)
end

end
