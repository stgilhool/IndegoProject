#script that generates data structures in order to make the transition matrix

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import os.path
import pdb
import code
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

startStatUn = np.unique(startStatInt)
endStatUn = np.unique(endStatInt)
#FIXME: make sure that startStatUn and endStatUn contain all the same stations

# vector of route identifiers (string concat of start + end)
routeIDtot = [startStat[i]+endStat[i] for i in range(len(startStat))]

#turn this vector into a numpy array and then into an integer array
routeID = np.array(routeIDtot)
# print(type(routeID))
routeID = routeID.astype(int)

#code.interact(local=locals())

primeQ = np.zeros([len(startStatUn), len(startStatUn)])

iter = 0
for rowIdx, startStation in enumerate(startStatUn):
    for colIdx, endStation in enumerate(startStatUn):
        #construct the id
        routeIDiter = str(startStation)+str(endStation)
        routeIDiterInt = int(routeIDiter)
        # get route indices for give 8-digit route number
        routeIndices = get_route_index(routeIDiterInt, route_data=routeIDtot)
        # Number of trips for given routeID
        ntripsRoute = len(routeIndices)

        # Store it in the big, unnormalized Q
        primeQ[rowIdx, colIdx] = ntripsRoute

        print(iter)
        iter = iter + 1

print(type(primeQ))
print(len(primeQ))

code.interact(local=locals())
