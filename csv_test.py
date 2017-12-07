import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import os.path
import pdb
import code



#Path and filename for data
filename = "indego-trips-2017-q3.csv"
#file_dir = "D:\\Users\\gilhool\\.atom\\storage\\Fuckit"

#Navigate to data file
#os.chdir(file_dir)

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

# Set of uniq identifiers
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

#code.interact(local=locals())

tripidx = 6
sample_start = s[tripidx]
sample_end = e[tripidx]

start_check = (s == sample_start)
end_check = (e == sample_end)

match_idx = []

#entryidx = 0
#for entry in starti:
#    if entry and endi[entryidx]:
#        matchi.append(entryidx)
#    entryidx = entryidx + 1


for i in range(len(start_check)):
    if start_check[i] and end_check[i]:
        match_idx.append(i)

durs = d[match_idx]

plt.hist(durs)
plt.show()

#code.interact(local=locals())
