from pickletools import optimize
import dataManager as dm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from keras.models import Input, Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model

def main():
    X = []
    Y = []
    print("tf version: ", tf.__version__)
    #for id in range(1025, 1026):

    isScalerExist = False
    window = 100 #based on last 10sec's data
    future = 1 #predict 100ms
    for id in range(1025, 1126, 1):
        print(id)
        df = dm.callKalTopDataFromFile(id)
        # HO = dm.callHODataFromFile(id)
        # print(HO.head())
        # dm.drawGraphHO(df, HO)

        if isScalerExist:
            Xs = s1.transform(df[["3_RSSI", "2_RSSI", "1_RSSI"]])
            Ys = s2.transform(df[["1_RSSI"]])
        else:
            s1 = MinMaxScaler(feature_range=(-1, 1))
            Xs = s1.fit_transform(df[["3_RSSI", "2_RSSI", "1_RSSI"]])
            s2 = MinMaxScaler(feature_range=(-1, 1))
            Ys = s2.fit_transform(df[["1_RSSI"]])
            isScalerExist = True

        for i in range(window, len(Xs)-future):
            if np.random.rand() > 0.990:
                X.append(Xs[i-window:i,:])
                Y.append(Ys[i+future])

    X, Y = np.array(X), np.array(Y)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

    es = EarlyStopping(monitor = 'loss', mode = 'min', verbose = 1, patience = 10)

    t0 = time.time()
    history = model.fit(X, Y, epochs = 1, batch_size = 1000, callbacks = [es], verbose = 1)
    t1 = time.time()
    print("Runtime: %.2f s" % (t1-t0))

    #copyModel = model
    model.save("./savemodel/model")
    # model.save("model.h5")
    # tf.saved_model.save(copyModel, "./")
    model = load_model("./savemodel/model")

    for id in range(3026, 3825):
        X = []
        Y = []
        Xraw = []
        print(id)
        df = dm.callKalTopDataFromFile(id)
        df2 = dm.callTopDataFromFile(id)

        Xs = s1.transform(df[["3_RSSI", "2_RSSI", "1_RSSI"]])
        Ys = s2.transform(df[["1_RSSI"]])
        Xraw = df2[["1_RSSI"]]

        for i in range(window, len(Xs)-future):
            X.append(Xs[i-window:i,:])
            Y.append(Ys[i+future])

        X, Y = np.array(X), np.array(Y)
        Xraw = np.array(Xraw)

        Yp = model.predict(X)
        Yu = s2.inverse_transform(Yp)
        Ym = s2.inverse_transform(Y)
        plt.figure(figsize=(10,6))
        plt.subplot(2, 1, 1)
        plt.plot(df["Time"][window+future:],Yu,'r-',label='LSTM')
        plt.plot(df["Time"][window+future:],Ym,'k--',label='KF')
        plt.plot(df["Time"][:],Xraw,'b-',label='rawRSSI')
        plt.legend()
        plt.show()

main()