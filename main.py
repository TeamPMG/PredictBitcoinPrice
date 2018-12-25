import numpy as np
import matplotlib.pyplot as plt
import csv

import chainer
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
from chainer import serializers

BATCH_SIZE = 256
NUM_EPOCH = 10000
INPUT_SIZE=128
OUTPUT_SIZE=16
TESTDATA_SIZE = 200

FILE_NAME="jpbitcoinbpi201811.csv"

class predict_model(chainer.Chain):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(predict_model,self).__init__(
            l1 = L.Linear(in_dim, in_dim),
            l2 = L.Linear(in_dim, out_dim),
            c1 = L.ConvolutionND(ndim=1, in_channels=1, out_channels=hidden_dim, ksize=3, pad=1),
            c2 = L.ConvolutionND(ndim=1, in_channels=hidden_dim, out_channels=hidden_dim, ksize=3, pad=1),
            c3 = L.ConvolutionND(ndim=1, in_channels=hidden_dim, out_channels=1, ksize=3, pad=1),
        )

    def __call__(self,x):
        h = F.relu(self.l1(x))
        h = F.reshape(h, (h.shape[0],1,h.shape[1]))
        h = F.relu(self.c1(h))
        h = F.dropout(h, 0.5)
        h = F.relu(self.c2(h))
        h = F.dropout(h, 0.5)
        h = F.relu(self.c3(h))
        h = F.dropout(h, 0.5)
        return self.l2(h)

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
        y = [float(y[4]) for y in data]
        #plt.plot(x,y)
        #plt.show()
        return y


def train():
    print('import data....')
    data = data_import()
    print('success!')
    data_x=[]
    data_y=[]
    for i in range(len(data) - INPUT_SIZE - OUTPUT_SIZE):
       data_x.append(data[i:i+INPUT_SIZE])
       data_y.append(data[i+INPUT_SIZE:i+INPUT_SIZE+OUTPUT_SIZE])

    data_x=np.array(data_x).astype("float32")
    data_y=np.array(data_y).astype("float32")

    X = data_x[0:len(data_x) - TESTDATA_SIZE]
    y = data_y[0:len(data_y) - TESTDATA_SIZE]
    test_x = data_x[len(data_x) - TESTDATA_SIZE:]
    test_y = data_y[len(data_y) - TESTDATA_SIZE:]

    mean=test_x.mean(axis=1)
    std=test_x.std(axis=1)
    test_x=(test_x-mean.reshape(TESTDATA_SIZE,1))/std.reshape(TESTDATA_SIZE,1)
    test_y=(test_y-mean.reshape(TESTDATA_SIZE,1))/std.reshape(TESTDATA_SIZE,1)
    model = predict_model(INPUT_SIZE, OUTPUT_SIZE,32)
    try:
        serializers.load_npz("predict.model", model)
        print("loaded")
    except:
        print("couldn't load")

    opt = chainer.optimizers.SGD(0.01)
    opt.setup(model)  
    num_batches = int(X.shape[0] / BATCH_SIZE)
    for epoch in range(NUM_EPOCH):
        perm = np.random.permutation(X.shape[0])
        for index in range(num_batches):
            X_batch = X[perm[index*BATCH_SIZE:(index+1)*BATCH_SIZE]]
            Y_batch = y[perm[index*BATCH_SIZE:(index+1)*BATCH_SIZE]]
            #return X_batch
            mean = X_batch.mean(axis=1)
            std = X_batch.std(axis=1)

            X_batch = (X_batch - mean.reshape(BATCH_SIZE,1))/std.reshape(BATCH_SIZE,1) + np.random.normal(0,0.5,X_batch.shape).astype(np.float32)
            Y_batch = (Y_batch - mean.reshape(BATCH_SIZE,1))/std.reshape(BATCH_SIZE,1) + np.random.normal(0,0.5,Y_batch.shape).astype(np.float32) 

            yl = model(X_batch)
            loss = F.mean_squared_error(yl,Y_batch)

            model.cleargrads()
            loss.backward()
            opt.update()

            chainer.config.train = False
            test_loss = F.mean_squared_error(model(test_x),test_y)
            chainer.config.train = True
            print("epoch:%d batch:%d/%d loss:%f test_loss:%f"%(epoch,index,num_batches,loss.data,test_loss.data))
            
        serializers.save_npz('predict.model', model)

def predict_price(input_data,output_data=None):
    chainer.config.train = False
    model = predict_model(INPUT_SIZE, OUTPUT_SIZE,32)
    try:
        serializers.load_npz("predict.model", model)
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
    predicted=model(data).data
    chainer.config.train = True
    
    y1=np.concatenate((y1, predicted.reshape(OUTPUT_SIZE,)*std+mean))
    y2=np.concatenate((y2, output_data))
    plt.plot(x,y1,label="predict")
    plt.plot(x,y2,label="real",linestyle="--")
    plt.show()

train()
    
