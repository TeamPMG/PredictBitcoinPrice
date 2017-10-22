import krakenex
import numpy as np
import matplotlib.pyplot as plt
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Activation,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Reshape
from keras.optimizers import Adam



PAIR="XXBTZJPY"
#key=input("Input Key>")
#secret=input("Input Secret>")
k = krakenex.API("key","secret")

def get_price():
    ticker=k.query_public("Ticker", {'pair':'XXBTZJPY'})
    Bid_Price = float(ticker["result"][PAIR]["a"][0])
    return Bid_Price


def predict_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=128))
    model.add(Activation('tanh'))
    model.add(Dense(256))
    model.add(Activation('tanh'))
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((512,1), input_shape=(512,)))
    model.add(Conv1D(1, 4, padding='same'))
    model.add(Activation('tanh'))
    model.add(Conv1D(1, 4, padding='same'))
    model.add(Activation('tanh'))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('tanh'))
    return model


    

k = krakenex.API("key","secret")

def train():
    x=np.arange(0,100,1)
    Prices=np.loadtxt('data.txt', delimiter='\t')
    
    p=predict_model()
    p_opt = Adam(lr=1e-5, beta_1=0.1)
    p.compile(loss='binary_crossentropy', optimizer=p_opt)
    
    for i in range(100):
        Now_Price=get_price()
        
        
        if Now_Price<Prices[99]:
            Up_Down=-1
        elif Now_Price>Prices[99]:
            Up_Down=1
        else:
            Up_Down=0
        #予測
        Predicted_Up_Down=p.predict(Prices.reshape(1,100))
        print(Predicted_Up_Down)
        print("predict: %f" % (Predicted_Up_Down[0][0]))
        print("real: %f" % (Up_Down))
        #学習
        p.train_on_batch(Prices.reshape(1,100), np.array([[Up_Down]]))
        
            
        Prices=np.append(Prices,Now_Price)
        Prices=np.delete(Prices,0)
           
        plt.plot(x,Prices)
        plt.show()
        p.save_weights('predict.h5')
        time.sleep(60)

    plt.plot(x,Prices)
    plt.show()


train()
    
