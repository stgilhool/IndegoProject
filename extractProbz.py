# ok. so, we switched it up. this script is going to attempt to take the data
#infrastructure lessons we learned in v1, and extract the number of trips from
#each station.

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

filename = "indego-trips-2017-q3.csv"
file_dir = "C:\\Users\Eric\\Desktop\\job_search\\data_science\\project\\data"

#Navigate to data file
os.chdir(file_dir)

#Initialize arrays
duration = []
trip_id = []
start_station = []
end_station = []

start_lat = []
start_lon = []

end_lat = []
end_lon = []

#Open CSV file
with open(filename) as testCSV:
    test1 = csv.reader(testCSV)

    #Loop through and read in data
    ii = 0
    for row in test1:
        if ii > 0:

            duration.append(row[1])
            trip_id.append(row[0])
            start_station.append(row[4])
            end_station.append(row[7])

            start_lat.append(row[5])
            start_lon.append(row[6])

            end_lat.append(row[8])
            end_lon.append(row[9])

        ii = ii + 1

#convert station lists from strings into list of integers
start_stationInt = list(map(int, start_station))
end_stationInt = list(map(int, end_station))

#convert the (x,y) coordinate pairs from strings into floats. This is done
#(maybe stupidly...) in a loop, since we need to figure out where the blank
#values are. To track the blank values, input 99999 into the resulting float
#array, and also keep track of the indices of the blanks
start_latFloat = []
startLatIdx_mask = []
for idx, lat in enumerate(start_lat):
    if lat == "":
        start_latFloat.append(float(99999))
        startLatIdx_mask.append(idx)
    else:
        start_latFloat.append(float(lat))

start_lonFloat = []
startLonIdx_mask = []
for idx, lon in enumerate(start_lon):
    if lon == "":
        start_lonFloat.append(float(99999))
        startLonIdx_mask.append(idx)
    else:
        start_lonFloat.append(float(lon))


end_latFloat = []
endLatIdx_mask = []
for idx, lat in enumerate(end_lat):
    if lat == "":
        end_latFloat.append(float(99999))
        endLatIdx_mask.append(idx)
    else:
        end_latFloat.append(float(lat))

end_lonFloat = []
endLonIdx_mask = []
for idx, lon in enumerate(end_lon):
    if lon == "":
        end_lonFloat.append(float(99999))
        endLonIdx_mask.append(idx)
    else:
        end_lonFloat.append(float(lon))

#start_latFloat = list(map(float, start_lat))
#start_lonFloat = list(map(float, start_lon))

#end_latInt = list(map(float, end_lat))
#end_lonInt = list(map(float, end_lon))

# Convert lists into np arrays, and then to int type
# find the number of unique entries in this array

# durarr = np.array(duration)
# d = durarr.astype(int)

startStat = np.array(start_station)
startStat = startStat.astype(int)

endStat = np.array(end_station)
endStat = endStat.astype(int)

startStatUn = np.unique(startStat)
endStatUn = np.unique(endStat)

# vector of route identifiers (string concat of start + end)
routeIDtot = [startStat[i]+endStat[i] for i in range(len(startStat))]

#turn this vector into a numpy array and then into an integer array
routeID = np.array(routeIDtot)
# print(type(routeID))
routeID = routeID.astype(int)

# now start writing the new script to extract the ID
startStation2 = []
endStation2 = []

startStation2.append([rIdx[0:4] for rIdx in route_id_set])
endStation2.append([rIdx[4:9] for rIdx in route_id_set])

startStation_set = list(set(start_station))
endStation_set = list(set(end_station))

startStation_setInt = list(map(int, startStation_set))
endStation_setInt = list(map(int, endStation_set))


# try the bincount command to return an array of frequencies
routeFreq = np.bincount(routeID)
#find non-zero elements
jj = np.nonzero(routeFreq)[0]

#use the zip command to create an array with unique route ID and frequency
# of occurence in the master table
routeIDfreqList = list(zip(jj,routeFreq[jj]))

routeIDfreq = np.array(routeIDfreqList)
routeIDreq = routeIDfreq.astype(int)




code.interact(local=locals())
