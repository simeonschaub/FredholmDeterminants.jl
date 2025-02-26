using TestItemRunner

@run_package_tests

@testitem "β = 1" begin
    using Distributions, QuadGK
    β, μ₀, σ²₀ = 1, -1.206533574582, 1.607781034581

    μ, _ = quadgk(x -> x * pdf(TracyWidom{β}(), x), -Inf, Inf)
    @test μ ≈ μ₀ rtol = 2.0e-14

    σ², _ = quadgk(x -> (x - μ₀)^2 * pdf(TracyWidom{β}(), x), -Inf, Inf)
    @test σ² ≈ σ²₀ rtol = 5.0e-13

    using ForwardDiff

    μ, _ = quadgk(x -> x * ForwardDiff.derivative(x -> cdf(TracyWidom{β}(), x), x), -10, 10)
    @test μ ≈ μ₀ rtol = 2.0e-10

    σ², _ = quadgk(x -> (x - μ₀)^2 * ForwardDiff.derivative(x -> cdf(TracyWidom{β}(), x), x), -10, 10)
    @test σ² ≈ σ²₀ rtol = 2.0e-9
end

@testitem "β = 2" begin
    using Distributions, QuadGK
    β, μ₀, σ²₀ = 2, -1.771086807411, 0.8131947928329

    μ, _ = quadgk(x -> x * pdf(TracyWidom{β}(), x), -Inf, Inf)
    @test μ ≈ μ₀ rtol = 4.0e-13

    σ², _ = quadgk(x -> (x - μ₀)^2 * pdf(TracyWidom{β}(), x), -Inf, Inf)
    @test σ² ≈ σ²₀ rtol = 2.0e-13

    using ForwardDiff

    μ, _ = quadgk(x -> x * ForwardDiff.derivative(x -> cdf(TracyWidom{β}(), x), x), -10, 10)
    @test μ ≈ μ₀ rtol = 4.0e-13

    σ², _ = quadgk(x -> (x - μ₀)^2 * ForwardDiff.derivative(x -> cdf(TracyWidom{β}(), x), x), -10, 10)
    @test σ² ≈ σ²₀ rtol = 2.0e-13
end

@testitem "β = 4" begin
    using Distributions, QuadGK
    β, μ₀, σ²₀ = 4, -2.306884893241, 0.5177237207726

    μ, _ = quadgk(x -> x * pdf(TracyWidom{β}(), x), -Inf, Inf)
    @test μ ≈ μ₀ rtol = 7.0e-9

    σ², _ = quadgk(x -> (x - μ₀)^2 * pdf(TracyWidom{β}(), x), -Inf, Inf)
    @test σ² ≈ σ²₀ rtol = 2.0e-8

    using ForwardDiff

    μ, _ = quadgk(x -> x * ForwardDiff.derivative(x -> cdf(TracyWidom{β}(), x), x), -5, 1)
    @test μ ≈ μ₀ rtol = 3.0e-5

    σ², _ = quadgk(x -> (x - μ₀)^2 * ForwardDiff.derivative(x -> cdf(TracyWidom{β}(), x), x), -5, 1)
    @test σ² ≈ σ²₀ rtol = 7.0e-4
end
