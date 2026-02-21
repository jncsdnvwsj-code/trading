function ∂montecarlo_call_price(S0, T, r, sigma, N)
    mc_average = 0.0
    for i in 1:N
        Z = randn()
        ST = S0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
        if ST > K
            mc_average += exp(-r * T) * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
        else 
            mc_average += 0.0
        end
    end
    return mc_average / N
end
S0, K, T, r, sigma, N = 10., 12.0, 3, 0.05, 0.13, 1_000_000



using ForwardDiff



mc_greek = ForwardDiff.derivative(x -> montecarlo_price(call_payoff, x, T, r, sigma, N), S0)
mc_greek2 = ∂montecarlo_call_price(S0, T, r, sigma, N)

bs_greek = ForwardDiff.derivative(x -> blackscholesprice(x, K, 0.0, T, r, sigma), S0)
Δ = cdf(Normal(), d1(S0, K, T, r, sigma))

bs_greek - Δ

mc_greek - Δ