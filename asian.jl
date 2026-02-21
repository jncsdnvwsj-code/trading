# Monte Carlo pricing for Asian options


function montecarlo_price_asian(payoff, S0, T, r, sigma, N)
    # Generate N random samples from a standard normal distribution
    mc_average = 0.0
    for i in 1:N
        Z = randn()
    
       
        # Simulate the stock price at maturity T
       
        S = simulate_geometric_brownian_motion_path(S0, r, sigma, T, 100)[2] 
        integralT  = sum(S) * T / length(S) / T # Average price over the path   
             
        # Discount the average payoff back to present value
        mc_average += exp(-r * T) * payoff(S[end], integralT)
    end
    return mc_average / N
end

mc_call_price_asian = montecarlo_price_asian((ST, integralT) -> max(integralT - K, 0), S0, T, r, sigma, N)