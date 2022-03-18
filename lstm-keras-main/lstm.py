from pickletools import optimize
import dataManager as dm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model

def main():
    datatypeIndex = 1
    np.random.seed(60)
        #for id in range(1025, 3824):
    for id in range(1025, 1026):
        df = dm.callDataFromFile(id, datatypeIndex)
        print(df.describe())
    rmvIndex = df[df['RSSI_5'] < -50].index
    df.drop(rmvIndex, inplace=True)

    s1 = MinMaxScaler(feature_range=(-1, 1))
    Xs = s1.fit_transform(df[["RSSI_5"]])

    s2 = MinMaxScaler(feature_range=(-1, 1))
    Ys = s2.fit_transform(df[["RSSI_5"]])

    window = 100

    X = []
    Y = []
    for i in range(window, len(Xs)):
        # print(Xs[i-window:i,:])
        # print(Ys[i])
        # print(i)
        X.append(Xs[i-window:i,:])
        Y.append(Ys[i])
    
    X, Y = np.array(X), np.array(Y)

    # print(len(X))
    # print(len(Y))
    # return 0

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
    history = model.fit(X, Y, epochs = 1, batch_size = 250, callbacks = [es], verbose = 1)
    t1 = time.time()
    print("Runtime: %.2f s" % (t1-t0))

    # plt.figure(figsize=(8,4))
    # plt.semilogy(history.history['loss'])
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.savefig('loss.png')
    model.save('model.h5')

    Yp = model.predict(X)
    Yu = s2.inverse_transform(Yp)
    Ym = s2.inverse_transform(Y)
    plt.figure(figsize=(10,6))
    plt.subplot(2, 1, 1)
    plt.plot(df["Time"][window:],Yu,'r-',label='LSTM')
    plt.plot(df["Time"][window:],Ym,'k--',label='Measured')
    plt.legend()
    plt.show()

    model = load_model('model.h5')
    test_df = dm.callDataFromFile(2026, datatypeIndex)

    rmvIndex = test_df[test_df['RSSI_5'] < -50].index
    test_df.drop(rmvIndex, inplace=True)
    
    Xt = test_df[["RSSI_5"]].values
    Yt = test_df[['RSSI_5']].values
    Xts = s1.transform(Xt)
    Yts = s2.transform(Yt)
    Xti = []
    Yti = []
    for i in range(window, len(Xts)):
        Xti.append(Xts[i-window:i,:])
        Yti.append(Yts[i])

    Xti, Yti = np.array(Xti), np.array(Yti)
    Ytp = model.predict(Xti)

    Ytu = s2.inverse_transform(Ytp)
    Ytm = s2.inverse_transform(Yti)
    plt.figure(figsize=(10,6))
    plt.subplot(2, 1, 1)
    plt.plot(test_df["Time"][window:],Ytu,'r-',label='LSTM')
    plt.plot(test_df["Time"][window:],Ytm,'k--',label='Measured')
    plt.legend()
    plt.show()

    Xtsq = Xts.copy()
    for i in range(1200, 1300):
        Xin = Xtsq[i-window:i].reshape((1, window, 1))
        Xtsq[i][0] = model.predict(Xin)
        Yti[i - window] = Xtsq[i][0]
        print(i)
    
    Ytu = s2.inverse_transform(Yti)
    plt.figure(figsize=(10,6))
    plt.subplot(2, 1, 1)
    plt.plot(test_df["Time"][window:],Ytu,'r-',label='LSTM')
    plt.plot(test_df["Time"][window:],Ytm,'k--',label='Measured')
    plt.legend()
    plt.show()

main()
