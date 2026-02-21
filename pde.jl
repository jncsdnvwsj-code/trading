# Black scholes PDE solver using finite difference method
using LinearAlgebra, Distributions

# Parameters
S0 = 95.0
T = 1.0        # Time to maturity
K = 100.0      # Strike price
r = 0.05       # Risk-free interest rate
σ = 0.2    # Volatility 
g(S) = max(S - K, 0)  # Payoff function for a call option

# Closed form solution for Black–Scholes price
using Distributions
d1(S, K, τ , r, σ) = (log(S / K) + (r + σ^2 / 2) * τ) / (σ * sqrt(τ))
d2(S, K, τ, r, σ) = d1(S, K, τ, r, σ) - σ * sqrt(τ)
function blackscholesprice(S, K, t, T, r, σ) 
    S * cdf(Normal(), d1(S, K, T - t, r, σ)) - K * exp(-r * (T - t)) * cdf(Normal(), d2(S, K, T - t, r, σ))
end





# Discretization parameters
S_max = 200.0  # Maximum stock price / roof
S_min = 0.0    # Minimum stock price
ts = range(0, T, length=40_001) # Time grid)
xs = range(S_min, S_max, length=1_001)
M = length(xs)
dx = (S_max - S_min) / (M - 1)
N = length(ts)
dt = T / (N - 1)

# Stability check for explicit Euler (diffusion-dominated constraint)
a_max = 0.5 * σ^2 * S_max^2
dt_max = dx^2 / (2 * a_max)   # ≈ dx^2 / (σ^2 * S_max^2)

if dt > dt_max
    println("Warning: dt = $dt > $dt_max (stability limit). The scheme is likely unstable.")
else
    println("dt = $dt <= $dt_max (diffusion stability limit satisfied).")
end




function ∂(v) 
    dv = [v[i+1] - v[i-1] for i in 2:(length(v)-1)]/(2*dx)
    dv = [dv[1]; dv; dv[end]] # padding boundary values
    return dv
end 

function ∂²(v) 
    d²v = [v[i+1] - 2*v[i] + v[i-1] for i in 2:(length(v)-1)]/(dx^2)
    d²v = [d²v[1]; d²v; d²v[end]] # padding boundary values
    return d²v
end


unitvector(M, i) = [j == i ? 1.0 : 0.0 for j in 1:M]
function function_as_matrix(f, M)
    fvectors = [f(unitvector(M, i)) for i in 1:M] # finite difference matrix for first derivative
    fmatrix = [fvectors[i][j] for j in 1:M, i in 1:M] # convert to matrix form
    return fmatrix
end
∂matrix = function_as_matrix(∂, M)
∂²matrix = function_as_matrix(∂², M)





# Time stepping backwards
# ∂v/∂t + r*S*∂v/∂S + 0.5*σ^2*S^2*∂²v/∂S² - r*v = 0
# V[i, j] ≈ v(t_i, S_j)

# (V[i+1, :] - V[i, :])/dt + map(S->r*S, xs).*∂(V[i+1, :]) + 0.5*σ^2*map(S->S^2, xs).*∂²(V[i+1, :]) - r*V[i+1, :] = 0

V = zeros(N, M)
V[end, :] = g.(xs) # terminal condition at maturity
using SparseArrays

L = sparse(r*Diagonal(xs)*∂matrix + 0.5*σ^2*Diagonal(xs.^2)*∂²matrix - r*I)
A = I + dt*L
MODE = :crank_nicolson # choose between :explicit, :matrix_explicit, :crank_nicolson

for i in (N-1):-1:1

    if MODE == :explicit
        # Solve for V[i, :]
        V[i, :] = V[i+1, :] + dt*(r*xs.*∂(V[i+1, :]) + 0.5*σ^2*map(S->S^2, xs).*∂²(V[i+1, :]) - r*V[i+1, :])
    elseif MODE == :matrix_explicit
        V[i, :] = A*V[i+1, :]
    elseif MODE == :crank_nicolson
        V[i, :] = ((I - 0.5*dt*L) \ ((I + 0.5*dt*L) * V[i+1, :]))
    end

    V[i, 1] = exp(-r*(T - ts[i])) * g(0.0) 
    V[i, end] = S_max - K*exp(-r*(T - ts[i])) # boundary condition at S=S_max from formula script
    # if g is not a call, you could use the following instead:
    gprime = (g(S_max) - g(S_max - dx)) / dx
    V[i, end] = gprime*S_max*exp(-r*(T - ts[i])) + (g(S_max) - gprime*S_max)*exp(-r*(T - ts[i]))
end

price1 = blackscholesprice(S0, K, 0.0, T, r, σ)
price2 = V[1, findfirst(x -> x >= S0, xs)]
println("Black–Scholes price (closed form): ", price1)
println("Black–Scholes price (PDE solver): ", price2)