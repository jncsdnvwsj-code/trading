# Monte Carlo estimator Black Scholes Call price 
using ForwardDiff
using Distributions

d1(S, K, τ , r, σ) = (log(S / K) + (r + σ^2 / 2) * τ) / (σ * sqrt(τ))

d2(S, K, τ, r, σ) = d1(S, K, τ, r, σ) - σ * sqrt(τ)

function blackscholesprice(S, K, t, T, r, σ) 
    S * cdf(Normal(), d1(S, K, T - t, r, σ)) - K * exp(-r * (T - t)) * cdf(Normal(), d2(S, K, T - t, r, σ))
end

function simulate_geometric_brownian_motion_path(S0, μ, σ, T, N)
    dt = T / N
    W = cumsum(sqrt(dt) * randn(N))
    t = 0:dt:T
    S = S0 * exp.((μ - σ^2 / 2) * t[1:end-1] + σ * W)
    return t[1:end-1], S
end 

function simulate_sde(S0, drift, diffusion, T, N)
    dt = T / N
    # Euler-Maruyama method
    S = zeros(N)
    S[1] = S0
    for i in 2:N
        dW = sqrt(dt) * randn()
        S[i] = S[i-1] + drift(S[i-1]) * dt + diffusion(S[i-1]) * dW
    end
    return S    
end

function montecarlo_price(payoff, S0, T, r, sigma, N)
    # Generate N random samples from a standard normal distribution
    mc_average = 0.0
    for i in 1:N
        Z = randn()
    
       
        # Simulate the stock price at maturity T
        #ST = S0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
       
        ST = simulate_geometric_brownian_motion_path(S0, r, sigma, T, 100)[2][end] # Simulate the stock price at maturity T

             
        # Discount the average payoff back to present value
        mc_average += exp(-r * T) * payoff(ST)
    end
    return mc_average / N
end

S0, K, T, r, sigma, N = 10., 12.0, 3, 0.05, 0.13, 1_000_000

call_payoff(ST) = max(ST - K, 0)
put_payoff(ST) = max(K - ST, 0)
    

mc_call_price = montecarlo_price(call_payoff, S0, T, r, sigma, N)

bs_call_price = blackscholesprice(S0, K, 0.0, T, r, sigma)

println("Monte Carlo Call Price: $mc_call_price")
println("Black-Scholes Call Price: $bs_call_price")
println("Difference: $(abs(mc_call_price - bs_call_price))")


mc_put_price = montecarlo_price(put_payoff, S0, T, r, sigma, N)

# Check the put-call parity
parity = mc_call_price - mc_put_price - (S0 - K * exp(-r * T))