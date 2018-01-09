import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import os.path
import pdb
import code

#note that this script runs after we have created the route ID matrix. for
#testing purposes, let's just copy and past the relevant code here. We'll
#need to go back through and delete this section later when we streamline
#everything

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
durarr = np.array(duration)
d = durarr.astype(int)

start_arr = np.array(start_station)
s = start_arr.astype(int)

end_arr = np.array(end_station)
e = end_arr.astype(int)

# vector of route identifiers (string concat of start + end)
route_id = [start_arr[i]+end_arr[i] for i in range(len(start_arr))]

# Set of unique identifiers
route_id_set = set(route_id)

# Route ID lookup table
route_table = []
for index, route in enumerate(route_id_set):
    route_table.append((index, route))
#    if index == 5:
        #print(route_table)

# now start writing the new script to extract the ID
startStation2 = []
endStation2 = []

startStation2.append([rIdx[0:4] for rIdx in route_id_set])
endStation2.append([rIdx[4:9] for rIdx in route_id_set])

startStation_set = list(set(start_station))
endStation_set = list(set(end_station))

startStation_setInt = list(map(int, startStation_set))
endStation_setInt = list(map(int, endStation_set))

startIdx = []
endIdx = []

#startLat = []
#startLon = []

#endLat = []
#endLon = []

#grab indices of first instance of each start and end station in master list
for Sstation in startStation_setInt:
    startIdx.append(next((i for i,v in enumerate(start_stationInt) if v == Sstation), None))
for Estation in endStation_setInt:
    endIdx.append(next((i for i,v in enumerate(end_stationInt) if v == Estation), None))

#grab unique (x,y) coords from master start/end (x,y) lists
#for idx in startIdx:
#    startLat.append(next((sLat for i,sLat in start_latInt if i == idx), None))
#    startLon.append(next((sLon for i,sLon in start_lonInt if i == idx), None))
#for idx in endIdx:
#    endLat.append(next((eLat for i,eLat in end_latInt if i == idx), None))
#    endLon.append(next((eLon for i,eLon in end_lonInt if i == idx), None))

#master list of unique route parameters for input into google API
#apiInput = []
#apiInput[0] = list(map(int, route_id_set)) #route hash
#apiInput[1] = list(map(int, startStation2)) #start stations
#apiInput[3] = list(map(int, endStation2)) #end stations

#apiInput[4] =  list(map(int, startLat))
#apiInput[5] =  list(map(int, startLon))
#apiInput[6] =  list(map(int, endLat))
#apiInput[7] =  list(map(int, endLon))


code.interact(local=locals())
