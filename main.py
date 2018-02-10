import numpy as np
import matplotlib.pyplot as plt
import csv

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Activation,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Reshape
from keras.optimizers import Adam
BATCH_SIZE = 10
NUM_EPOCH = 100
INPUT_SIZE=128
OUTPUT_SIZE=16
TESTDATA_SIZE=200

FILE_NAME="jpbitcoinbpi201801.csv"

def predict_model():
    model = Sequential()
    model.add(Dense(input_dim=128, output_dim=128))
    model.add(Activation('relu'))
    model.add(Reshape((128,1), input_shape=(128,)))
    model.add(Conv1D(256, 4,strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 4,strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(64, 4,strides=2, padding='same'))#64
    model.add(Activation('relu'))
    model.add(Conv1D(32, 4,strides=2, padding='same'))#32
    model.add(Activation('relu'))
    model.add(Conv1D(16, 4,strides=2, padding='same'))#16
    model.add(Activation('relu'))
    model.add(Conv1D(1, 1,strides=1, padding='same'))#16
    model.add(Activation('linear'))
    model.add(Reshape((16,), input_shape=(16,1)))
    
    
    
    return model

def data_import():
    
    with open(FILE_NAME, 'r') as f:
        reader = csv.reader(f)
        data=[]
        for row in reader:
            data.append(row)
            #date,open,high,low,close,volume
        #x = [x[0] for x in data]
        data=data[1:]
        
        x = [i for i in range(len(data))]
        y = [y[4] for y in data]
        plt.plot(x,y)
        plt.show()
    return y


def train():
    data = data_import()
    X=[]
    y=[]
    for i in range(len(data)-(INPUT_SIZE+OUTPUT_SIZE+TESTDATA_SIZE)):
       X.append(data[i:i+INPUT_SIZE])
       y.append(data[i+INPUT_SIZE:i+INPUT_SIZE+OUTPUT_SIZE])
    X=np.array(X).astype("float32")
    y=np.array(y).astype("float32")
        

    
    predict=predict_model()
    opt = Adam(lr=1e-4, beta_1=0.5)
    predict.compile(loss='mean_squared_error', optimizer=opt)
    
    
    num_batches = int(X.shape[0] / BATCH_SIZE)
    for epoch in range(NUM_EPOCH):
        perm = np.random.permutation(X.shape[0])
        for index in range(num_batches):
            X_batch = X[perm[index*BATCH_SIZE:(index+1)*BATCH_SIZE]]
            y_batch = y[perm[index*BATCH_SIZE:(index+1)*BATCH_SIZE]]
            mean = X_batch.mean()
            std = X_batch.std()

            X_batch = (X_batch - mean)/std
            y_batch = (y_batch - mean)/std

            predicted=predict.predict(X_batch)
            time = [i for i in range(OUTPUT_SIZE)]
            
            plt.plot(time,predicted[0],label="predict")
            plt.plot(time,y_batch[0],label="real")
            plt.show()
            
            loss = predict.train_on_batch(X_batch,y_batch)
            print("epoch:%d batch:%d/%d loss:%f"%(epoch,index,num_batches,loss))
        predict.save_weights('predict.h5')


train()
    
