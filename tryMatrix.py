#script that makes transition matrix and propagates trips

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import os.path
import pdb
import code
import time
import networkx as nx
from show_route_info import get_route_index

filename = "indego-trips-2017-q3.csv"
#file_dir = "C:\\Users\Eric\\Desktop\\job_search\\data_science\\project\\data"
file_dir = filename

#navigate to data file
#os.chdir(file_dir)

#initialize arrays
trip_id = []
start_station = []
end_station = []


#open CSV file; loop through the file and import data into the selected arrays
with open(filename) as testCSV:
    test1 = csv.reader(testCSV)
    ii = 0
    for row in test1:
        if ii > 0:

            trip_id.append(row[0])
            start_station.append(row[4])
            end_station.append(row[7])

        ii = ii + 1

#convert station lists from strings into list of integers
start_stationInt = list(map(int, start_station))
end_stationInt = list(map(int, end_station))

# Convert lists into np arrays, and then to int type
# find the number of unique entries in this array

startStat = np.array(start_station)
startStatInt = startStat.astype(int)

endStat = np.array(end_station)
endStatInt = endStat.astype(int)

# Get vector of unique station ID's
startStatUn = np.unique(startStatInt)
endStatUn = np.unique(endStatInt)

# Make sure that startStatUn and endStatUn contain all the same stations
stationCheck = np.array_equal(startStatUn, endStatUn)
if not stationCheck:
    sys.exit("ERROR: start and end station vectors contain different stations")
    

# vector of route identifiers (string concat of start + end)
routeIDtot = [startStat[i]+endStat[i] for i in range(len(startStat))]

#turn this vector into a numpy array and then into an integer array
routeID = np.array(routeIDtot)
# print(type(routeID))
routeID = routeID.astype(int)

#code.interact(local=locals())

primeQ = np.zeros([len(startStatUn), len(startStatUn)])

for rowIdx in range(len(startStat)):
    # Get Q row index
    startStatRow = startStatInt[rowIdx]
    rowIdxQ = np.where(startStatUn == startStatRow)

    endStatRow = endStatInt[rowIdx]
    colIdxQ = np.where(startStatUn == endStatRow)

    primeQ[rowIdxQ, colIdxQ] = primeQ[rowIdxQ, colIdxQ] + 1

matrixQ = primeQ/(primeQ.sum(axis=1, keepdims=True))

print(primeQ.sum(axis=1))
print(primeQ.sum(axis=0))
print(type(primeQ))
print(len(primeQ))

#bikevec = np.zeros(len(startStatUn))
bikevec = np.ones(len(startStatUn))
#bikevec[1] = 1.0

trip = np.matmul(bikevec, matrixQ)
print(trip)

trip2 = np.matmul(trip, matrixQ)
print(trip2)

fig = plt.figure(figsize=(12,9))
plt.ion()
plt.show()

for tripnum in range(0,20):
        
    if tripnum > 0:
        result = np.matmul(bikevec, matrixQ)
        bikevec = result

    elif tripnum == 0:
        result = bikevec
        
    trip = fig.add_subplot(111)
    titleString = 'Trip '+str(tripnum)
    trip.set_title(titleString)
    trip.set_autoscaley_on(False)
    trip.set_ylim([0,5])
    trip.plot(result, linestyle='None', marker='.')
    
    plt.draw()
    
    if tripnum == 0:
        plt.pause(3)
    else:
        plt.pause(1)
    
    fig.clf()


    

#code.interact(local=locals())
