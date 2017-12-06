import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import os.path
import pdb
import code


filename = "indego-trips-2017-q3.csv"
file_dir = "C:\\Users\\gilhool\\.atom\\storage\\Fuckit"

os.chdir(file_dir)
print(os.getcwd())
print(filename)
print(os.listdir(file_dir))

duration = []
trip_id = []
start_station = []
end_station = []

with open(filename) as testCSV:
    test1 = csv.reader(testCSV)
    ii = 0
    for row in test1:
        if ii > 0:
            duration.append(row[1])
            trip_id.append(row[0])
            start_station.append(row[4])
            end_station.append(row[7])
        ii = ii + 1

dd = np.array(duration)
d = dd.astype(int)

ss = np.array(start_station)
s = ss.astype(int)

ee = np.array(end_station)
e = ee.astype(int)

# vector of route identifiers (string concat of start+end)
#route_id = np.transpose(np.transpose(ss)+np.transpose(ee))
route_id = [ss[i]+ee[i] for i in range(len(ss))]

route_id_set = set(route_id)

for route in route_id_set:
    print("steve is on route: {0}".format(route))



#print(route_id[0:5])
code.interact(local=locals())



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
