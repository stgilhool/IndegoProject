import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import os.path
import pdb
import code

#this script takes the output of the csv loader and finds the coordinates
#of the start station and end station for each individual route_id

#copy of csv_test.py because i suck at python
#Path and filename for data
filename = "indego-trips-2017-q3.csv"
file_dir = "C:\\Users\Eric\\Desktop\\job_search\\data_science\\project\\data"

#Navigate to data file
os.chdir(file_dir)

#Initialize arrays
duration = []
trip_id = []
start_station = []
end_station = []

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
        ii = ii + 1

# Convert lists into np arrays, and then to int type
durarr = np.array(duration)
d = durarr.astype(int)

start_arr = np.array(start_station)
s = start_arr.astype(int)

end_arr = np.array(end_station)
e = end_arr.astype(int)

# vector of route identifiers (string concat of start+end)
#route_id = np.transpose(np.transpose(ss)+np.transpose(ee))
route_id = [start_arr[i]+end_arr[i] for i in range(len(start_arr))]

# Set of unique identifiers
route_id_set = set(route_id)

#for i in enumerate(route_id_set):
#    print(i)

#for i in route_id_set:
#    print(i)

# Route ID lookup table
route_table = []
for index, route in enumerate(route_id_set):
    route_table.append((index, route))
    if index == 5:
        print(route_table)


#first check if the length of each route id is the same by converting to a string
routeChk = []
for route in route_id_set:
        routeChk.append(route)


digitChk = []
rtChkLst = []
for i in range(len(routeChk)):
        digitChk.append(len(routeChk[i]))
        rtChkLst.append(str(routeChk[i]))

digitCmbo = set(digitChk)

print(digitCmbo)

#now that we are sure that the vectors are all the same length, we need to
#parse the route IDs to separate the first four digits from the last 4 digits

#sStation =[]
#eStation = []

#for i in range(len(routeChk)):



#code.interact(local=locals())
