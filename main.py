import numpy as np
import matplotlib.pyplot as plt
import csv

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers.core import Activation,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Reshape
from keras.optimizers import Adam
BATCH_SIZE = 10
NUM_EPOCH = 10000
INPUT_SIZE=128
OUTPUT_SIZE=16
TESTDATA_SIZE=200

FILE_NAME="jpbitcoinbpi201801.csv"

def predict_model():
    model = Sequential()
    model.add(Dense(input_dim=128, output_dim=128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Reshape((128,1), input_shape=(128,)))
    model.add(Conv1D(256, 4,strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(256, 4,strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(256, 4,strides=2, padding='same'))#64
    model.add(Activation('relu'))
    model.add(Conv1D(256, 4,strides=2, padding='same'))#32
    model.add(Activation('relu'))
    model.add(Conv1D(256, 4,strides=2, padding='same'))#16
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
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
    data=np.array(data).astype("float32")
    test_x=data[1800:1928].reshape(1,128)
    test_y=data[1928:1944].reshape(1,16)
    mean=test_x.mean()
    std=test_x.std()
    test_x=(test_x-mean)/std
    test_y=(test_y-mean)/std
        

    
    predict=predict_model()
    try:
        predict.load_weights('predict.h5')
        print("loaded")
    except:
        print("couldn't load")
    opt = Adam(lr=1e-4, beta_1=0.5)
    predict.compile(loss='mean_squared_error', optimizer=opt)
    
    
    num_batches = int(X.shape[0] / BATCH_SIZE)
    for epoch in range(NUM_EPOCH):
        perm = np.random.permutation(X.shape[0])
        for index in range(num_batches):
            X_batch = X[perm[index*BATCH_SIZE:(index+1)*BATCH_SIZE]]
            y_batch = y[perm[index*BATCH_SIZE:(index+1)*BATCH_SIZE]]
            #return X_batch
            mean = X_batch.mean(axis=1)
            std = X_batch.std(axis=1)

            X_batch = (X_batch - mean.reshape(BATCH_SIZE,1))/std.reshape(BATCH_SIZE,1)
            y_batch = (y_batch - mean.reshape(BATCH_SIZE,1))/std.reshape(BATCH_SIZE,1)

           
            loss = predict.train_on_batch(X_batch,y_batch)
            test_loss = predict.test_on_batch(test_x,test_y)
            print("epoch:%d batch:%d/%d loss:%f test_loss%f"%(epoch,index,num_batches,loss,test_loss))
        predicted=predict.predict(X_batch)
        time = [i for i in range(OUTPUT_SIZE)]
            
        plt.plot(time,predicted[0],label="predict",linestyle="--")
        plt.plot(time,y_batch[0],label="real")
        plt.show()
            
        predict.save_weights('predict.h5')

def predict_price(input_data,output_data=None):
    predict=predict_model()
    try:
        predict.load_weights('predict.h5')
        print("loaded")
    except:
        print("couldn't load")
    x=[i for i in range(INPUT_SIZE+OUTPUT_SIZE)]
    data=np.array(input_data).astype("float32")
    y1 = y2 = data
    mean = data.mean()
    std = data.std()
    data = (data-mean)/std
    
    data=data.reshape(1,INPUT_SIZE)
    predicted=predict.predict(data)
    
    y1=np.concatenate((y1, predicted.reshape(OUTPUT_SIZE,)*std+mean))
    y2=np.concatenate((y2, output_data))
    
    plt.plot(x,y1,label="predict",linestyle="--")
    plt.plot(x,y2,label="real")
    plt.show()
#train()
    
