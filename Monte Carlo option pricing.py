import numpy as np
from numpy import exp, sqrt, log, mean,std
import matplotlib.pyplot as plt

S_0 = 100
r = 0.2
vol = 0.3
T = 0.86
K = 130

N = 365
dt = T/N

def stock_price_path():
    path = [S_0]
    for i in range(0, N):
        Z = np.random.standard_normal()
        path.append(path[i] * exp((r - vol**2 / 2) * dt + vol * sqrt(dt) * Z))
    return path


M = 10000
plt.figure(figsize=(10, 6))

for _ in range(M):
    path = stock_price_path()
    t = np.linspace(0, T, N + 1)
    plt.plot(t, path, linewidth=1)

plt.title(f'Simulated stock price paths')
plt.xlabel('Time (years)')
plt.ylabel('Stock price ($)')
plt.grid(True)
plt.show()

def asian_call(path):
    return max((mean(path)-K),0)

def asian_put(path):
    return max((K-mean(path)),0)

def MC_asian_call(n):
    payoff=[]
    for _ in range(n):
        payoff.append(asian_call(stock_price_path()))
    return exp(-r*T)*mean(payoff)

print(MC_asian_call(100))