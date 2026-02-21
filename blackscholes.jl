using Distributions
using Downloads
using DelimitedFiles
using MarketData
using Dates


"""
    business_days(start_date, end_date)

Counts business days (Mon–Fri) between start_date (exclusive)
and end_date (inclusive).
"""
function business_days(start_date::Date, end_date::Date)
    start_date < end_date || return 0
    count = 0
    d = start_date + Day(1)
    while d <= end_date
        if dayofweek(d) ≤ 5   # 1=Mon, ..., 5=Fri
            count += 1
        end
        d += Day(1)
    end
    return count
end

"""
    load_yahoo_symbol(symbol; startdate=nothing, enddate=today())

Downloads daily data from Yahoo via MarketData.jl
and returns:

    ts :: Vector{Float64}   # time in years since first observation
    Ss :: Vector{Float64}   # closing prices

"""
function load_yahoo_symbol(symbol::String; startdate=nothing, enddate=today())

    # Download
    ta = startdate === nothing ?
        yahoo(Symbol(symbol), YahooOpt(period2 = DateTime(enddate))) :
        yahoo(Symbol(symbol), YahooOpt(period1 = DateTime(startdate)), period2 = DateTime(enddate))

    # Extract dates and closing prices
    dates = Date.(timestamp(ta))
    closes = values(ta)[:, findfirst(==(Symbol("Close")), colnames(ta))]

    Ss = Float64.(closes)

    return dates, Ss
end


d1(S, K, τ , r, σ) = (log(S / K) + (r + σ^2 / 2) * τ) / (σ * sqrt(τ))

d2(S, K, τ, r, σ) = d1(S, K, τ, r, σ) - σ * sqrt(τ)

function blackscholesprice(S, K, t, T, r, σ) 
    S * cdf(Normal(), d1(S, K, T - t, r, σ)) - K * exp(-r * (T - t)) * cdf(Normal(), d2(S, K, T - t, r, σ))
end

quadvar_logprice(S, σ, T) = σ^2 * T

function estimate_quadraticvariation(ts, Ss)
    logreturns = diff(log.(Ss))
    return sum(logreturns.^2)
end
    
function estimate_sigma(ts, Ss)
    T = ts[end] - ts[1]
    return sqrt(estimate_quadraticvariation(ts, Ss) / T)
end

function simulate_geometric_brownian_motion_path(S0, μ, σ, T, N)
    dt = T / N
    W = cumsum(sqrt(dt) * randn(N))
    t = 0:dt:T
    S = S0 * exp.((μ - σ^2 / 2) * t[1:end-1] + σ * W)
    return t[1:end-1], S
end 


S0 = 100.0
μ = 0.07    
σ = 0.2
T = 1.0
N = 1_000_000
r = 0.05

ts, Ss = simulate_geometric_brownian_motion_path(S0, μ, σ, T, N)
estimated_sigma = estimate_sigma(ts, Ss)
println("Estimated σ: ", estimated_sigma)


K = 100.0
# blackscholesprice(S, K, t, T, r, σ)
price = blackscholesprice(S0, K, 0.0, T, r, σ)
println("Black–Scholes price: ", price) 









# Downliad SPY
ds, Ss = load_yahoo_symbol("SPY"; enddate=Date("2026-02-11"))
Ss = Ss[end-100+1:end] # last 100 observations

ts = (0:length(Ss) .- 1)/252 # time in years since first observation, assuming 252 trading days per year

estimated_sigma = estimate_sigma(ts, Ss)

println("Estimated σ for SPY (annualized): ", estimated_sigma)


#= 
Real call price, looked up on February 10:
Instrument: SPY February 24, 2026 700 Call
Option Symbol: SPY260224C00700000

Underlying: SPY
Expiration Date: February 24, 2026
Strike Price: 700
Option Type: Call
Implied Volatility	11.65%

Last traded price: 3.46
=#

# Compare with Black–Scholes price for the same parameters:
S0 = Ss[end]
K = 700.0     
T = business_days(Date("2026-02-10"), Date("2026-02-24"))/252 # time to expiration in years (13 trading days) 
r = 0.0175      # risk-free rate 


price2 = blackscholesprice(S0, K, 0.0, T, r, estimated_sigma)



