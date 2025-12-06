import numpy as np
from numpy.random import randint as flip 
from numpy import exp, sqrt, log, mean,std
import matplotlib.pyplot as plt
from scipy.stats import norm

def phi(x):
    return norm.cdf(x)

N = 365
n = 1000
K = float(input("Enter the value of the strike price of the European call: "))
T = float(input("Enter the maturity in years of the European call: "))
S = float(input("Enter the initial stock price: "))
vol = float(input("Enter the volatility of the stock: "))
r = float(input("Enter the rate of return of the stock: "))

def stock_path(S, vol, r, T, N):
    stock_path=[S]
    h=T/N
    W=2*flip(2,size=N)-1
    for i in range(N):
        stock_path.append(S*exp((r-vol**2 / 2)*h*(i+1)+vol*sqrt(h)*sum(W[:i+1])))
    return stock_path

def european_call(stock_path, K):
    return max(stock_path[N]-K,0)

def d_1(S, K, vol, r, T):   
    return (log(S/K) + (r + vol**2 / 2)*T)/(vol * sqrt(T))

def d_2(S, K, vol, r, T):
    return d_1(S, K, vol, r, T) - vol * sqrt(T)

def BS_european_price(S, K, vol, r, T):
    return S*phi(d_1(S, K, vol, r, T)) - K * exp(-r*T)*phi(d_2(S, K, vol, r, T))

def montecarlo(S, vol, r, T,K, N,n):
    Payoff=[]
    for _ in range(n):
        Payoff.append(european_call(stock_path(S, vol, r, T, N),K))
    return[exp(-r*T)*mean(Payoff), 1.96*std(Payoff)/sqrt(n)]

def delta(S, K, vol, r, T):
    return phi(d_1(S, K, vol, r, T))

def gamma(S, K, vol, r, T):
    return delta(S, K, vol, r, T)/(S * vol * sqrt(T))

def rho(S, K, vol, r, T):
    return K * T * exp(-r*T)*delta(S, K, vol, r, T)

def theta(S, K, vol, r, T):
    return - S * delta(S, K, vol, r, T) * vol / (2 * sqrt(T)) - r*K*exp(-r*T)*delta(S, K, vol, r, T)

def vega(S, K, vol, r, T):
    return S*delta(S, K, vol, r, T)*sqrt(T)

print(BS_european_price(S, K, vol, r, T))
print(montecarlo(S, vol, r, T,K, N, n))