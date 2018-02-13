# this script sets up the infrastructure to visualize the network using networkx

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

# Convert lists into np arrays, and then to int type
durarr = np.array(duration)
d = durarr.astype(int)

start_arr = np.array(start_station)
startID = start_arr.astype(int)
startSet = set(start_stationInt)

end_arr = np.array(end_station)
endID = end_arr.astype(int)
endSet = set(end_stationInt)

#try using the 'where' command to find all instances of, for example, the first
# start station
startEx = startID[0]
testEx = np.where(startID == startEx)[0]

# try extracting end stations using the test example indices
endEx = endID[testEx]

# ok great. now we have to scale this up!

code.interact(local=locals())
