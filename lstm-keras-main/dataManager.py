import matplotlib.pyplot as plt
import pandas as pd
import os

def drawGraph(df):
    df.plot(x="Time", )
    plt.ylim([-100, 100])
    plt.legend(loc='lower right')
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def drawGraphHO(df, HO):
    df.plot(x="Time", )
    plt.ylim([-100, 100])
    for time in HO["Time"]:
        print(time)
        plt.axvline(x=time, color='r', linewidth=1)
    plt.legend(loc='lower right')
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def callHODataFromFile(id):
    str = "001/UE_%d.hot.csv" % id
    df = pd.read_csv(str, header=None)
    #print(df.isnull().sum())
    for i in range(1, 3):
        df = df.drop(df.columns[1], axis='columns')
        #print(df.head())
    df.columns = ["Time"]
    #print(df.head())
    return df

def callDataFromFile(id, type):
    enum = ['raw', 'kf']
    str = "001/UE_%d.%s.csv" % (id, enum[type])
    df = pd.read_csv(str, header=None)
    #print(df.isnull().sum())
    for i in range(1, 11):
        df = df.drop(df.columns[i], axis='columns')
        #print(df.head())
    df.columns = ["Time", "RSSI_0", "RSSI_1", "RSSI_2", "RSSI_3", "RSSI_4", "RSSI_5", "RSSI_6", "RSSI_7", "RSSI_8", "RSSI_9"]
    #print(df.head())
    return df

def callKalTopDataFromFile(id):
    str = "001U/UE_%d.kalmax.csv" % id
    df = pd.read_csv(str, header=None)
    df.columns = ["Time", "5_RSSI", "4_RSSI", "3_RSSI", "2_RSSI", "1_RSSI"]
    return df

def callTopDataFromFile(id):
    str = "001U/UE_%d.max.csv" % id
    df = pd.read_csv(str, header=None)
    df.columns = ["Time", "5_RSSI", "4_RSSI", "3_RSSI", "2_RSSI", "1_RSSI"]
    return df

def callAllDataUnderPath(path_dir):
    dfList = []
    fileList = os.listdir(path_dir)
    for filename in fileList:
        str = "%s/%s" % (path_dir, filename)
        df = pd.read_csv(str, header=None)
        df.columns = ["Time", "5_RSSI", "4_RSSI", "3_RSSI", "2_RSSI", "1_RSSI"]
        print(df.head())
        dfList.append(df)
    return dfList
