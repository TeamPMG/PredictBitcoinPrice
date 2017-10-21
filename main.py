import krakenex
import numpy as np
import matplotlib.pyplot as plt
import time
PAIR="XXBTZJPY"
#key=input("Input Key>")
#secret=input("Input Secret>")
k = krakenex.API("key","secret")

def get_price():
    ticker=k.query_public("Ticker", {'pair':'XXBTZJPY'})
    Bid_Price = float(ticker["result"][PAIR]["a"][0])
    return Bid_Price
    

k = krakenex.API("key","secret")
x=np.arange(0,100,1)
Prices=np.zeros(100)
Now_Price=get_price()
Prices=Prices+Now_Price


for i in range(100):
    Now_Price=get_price()
    Prices=np.append(Prices,Now_Price)
    Prices=np.delete(Prices,0)
    print(Now_Price)
    time.sleep(60)

plt.plot(x,Prices)
plt.show()