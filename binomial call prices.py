from math import sqrt
import numpy
from numpy import zeros, exp

#Setup few parameters
def FindParameters(p,alpha,sigma,T,N):
    h=T/N
    u=alpha*h+sigma*sqrt(h)*sqrt((1-p)/p)
    d=alpha*h-sigma*sqrt(h)*sqrt((1-p)/p)
    return [h,u,d]  
  
#Define binomial stock price
def BinomialStock(p,alpha,sigma,S0,T,N):
    [h,u,d]=FindParameters(p,alpha,sigma,T,N)
    TimePoints=zeros(N+1)
    StockPrices=zeros((N+1, N+1))
    StockPrices[0,0]=S0
    for i in range(1,N+1):
        TimePoints[i]=i*h
        StockPrices[i]=StockPrices[i-1]*exp(u)
        StockPrices[i,i]=StockPrices[i-1,i-1]*exp(d)
    return [TimePoints,StockPrices.T]
    
#[TimePoints,StockPrices]=BinomialStock(0.5,0.01,0.2,10,1/12,1000)
#print("TimePoints:\n",TimePoints)
#print("StockPrices:\n", StockPrices)

#Define few payoffs
def StandardCall(S,K):
    return max(S-K,0)
def StandardPut(S,K):
    return max(K-S,0)
def AsianCall(S,N,K): # By S we now denote the vector of historical stock prices
    return max(1/(N+1)*sum(S)-K,0)
def AsianPut(S,N,K): # By S we now denote the vector of historical stock prices
    return max(K-1/(N+1)*sum(S),0)


def RiskFreeProbability(r,h,u,d):
    q_u=(exp(r*h)-exp(d))/(exp(u)-exp(d))
    q_d=(exp(u)-exp(r*h))/(exp(u)-exp(d))
    return [q_u,q_d]

#Define binomial European call price
def BinomialEuropean(p,alpha,sigma,r,S0,K,T,N,OptionType):
    [h,u,d]=FindParameters(p,alpha,sigma,T,N)
    [q_u,q_d]=RiskFreeProbability(r,h,u,d)
    [TimePoints,StockPrices]=BinomialStock(p,alpha,sigma,S0,T,N)
    Payoff=numpy.array([[OptionType(StockPrices[j,i],K) for i in range(N+1)]  for j in range(N+1)])
    Price=zeros((N+1,N+1))
    for i in range(N+1):
        Price[i,N]=Payoff[i,N]
    for j in range(N):
        for i in range(N-j):
            Price[i,N-j-1]=exp(-r*h)*(q_u * Price[i,N-j]+q_d * Price[i+1,N-j])
    return(Price[0, 1])

print("Binomial European fair price at t=0:\n",BinomialEuropean(0.5,0.01,0.2,0.01,10,10,1/12,1000,StandardCall))

#Define binomial American call price
def BinomialAmerican(p,alpha,sigma,r,S0,K,T,N,OptionType):
    [h,u,d]=FindParameters(p,alpha,sigma,T,N)
    [q_u,q_d]=RiskFreeProbability(r,h,u,d)
    [TimePoints,StockPrices]=BinomialStock(p,alpha,sigma,S0,T,N)
    PayOff=zeros((N+1,N+1))
    for j in range(N+1):
        for i in range(j+1):
            PayOff[i,j]=OptionType(StockPrices[i,j],K) 
    Price=zeros((N+1,N+1))
    CashFlow=zeros((N,N))
    for i in range(N+1):
        Price[i,N]=PayOff[i,N]
    for j in range(N):
        for i in range(N-j):
            Price[i,N-j-1]=max(PayOff[i,N-j-1],exp(-r*h)*(q_u * Price[i,N-j]+q_d * Price[i+1,N-j]))
            CashFlow[i,N-j-1]=max(PayOff[i,N-j-1]-exp(-r*h)*(q_u * Price[i,N-j]+q_d * Price[i+1,N-j]),0)
    return [Price,CashFlow]

#[Price,CashFlow]=BinomialAmerican(0.5,0.01,0.2,0.01,10,10,1/12,5,StandardPut)
#print("Binomial American price:\n",Price)
#print("Cash flow:\n", CashFlow)

def PayOffVector(StockPrices,N,K, OptionType):
    PayOff=[]
    for i in range(2**N):
        HistoricalStock=[]
        downs=0
        index=i
        for j in range(N+1):
            HistoricalStock.append(StockPrices[downs][j])
            if index>=2**(N-1-j):
                index-=2**(N-1-j)
                downs+=1
        PayOff.append(OptionType(HistoricalStock,N,K))
    return PayOff

def reducePriceVector(PriceVector,qu,qd,r,h):
    NewVector=[]
    for j in range(len(PriceVector)//2):
        NewVector.append(exp(-r*h)*(qu*PriceVector[2*j]+qd*PriceVector[2*j+1]))
    return NewVector    

#Define binomial Asian call price
def BinomialAsianDirect(p,alpha,sigma,r,S0,K,T,N,OptionType):
    [h,u,d]=FindParameters(p,alpha,sigma,T,N)
    [q_u,q_d]=RiskFreeProbability(r,h,u,d)
    [TimePoints,StockPrices]=BinomialStock(p,alpha,sigma,S0,T,N)
    PayOff=PayOffVector(StockPrices,N,K, OptionType)
    PriceVector=PayOff
    for j in range(N):
        PriceVector=reducePriceVector(PriceVector,q_u,q_d,r,h)
    return(PriceVector[0])
    
#print("Binomial Asian price:",BinomialAsianDirect(0.5,0.01,0.2,0.01,10,10,1/12,5,AsianCall))