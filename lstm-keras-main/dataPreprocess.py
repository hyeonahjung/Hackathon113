import dataManager as dm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    datatypeIndex = 1
    for id in range(1025, 3825):
        df = dm.callDataFromFile(id, datatypeIndex)
        print(id)
        #dm.drawGraph(df)

        if(df["RSSI_0"][0] > -100):
            flag = True
        else:
            flag = False

        timeS = df["Time"]
        df = df.drop(columns = ["Time"])

        maxS = df.max(axis=1)

        newf = pd.DataFrame(np.sort(df.values)[:,-5:], columns=["5_RSSI", "4_RSSI", "3_RSSI", "2_RSSI", "1_RSSI"])
        print(newf.head())
        df = pd.concat([timeS, newf], axis=1)
        df.columns = ["Time", "5_RSSI", "4_RSSI", "3_RSSI", "2_RSSI", "1_RSSI"]
        #dm.drawGraph(df)

        if(flag):
            df.to_csv("001U/UE_%d.kalmax.csv" % id, index=False, header=False)
        else:
            df.to_csv("001U/UE_%d.kalmax.csv" % id, index=False, header=False)

main()