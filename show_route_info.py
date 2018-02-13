import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import os.path
import pdb
import code

### Function that takes route_id and finds all row indices that match
def get_route_index(route_id, route_data):
    
    # Make a boolean vector of matching/non-matching extries
    route_id_np = np.array(route_data)
    route_id_test = route_id_np.astype(int)

    route_id_full_test = int(route_id_full)


    #route_id_match_arr = (route_id_arr == route_id_full)
    route_id_match_arr = (route_id_test == route_id_full_test)

    #print(type(route_id_test))
    #print(type(route_id_full_test))
    #print(type(route_id_match_arr))
    #print(route_id_match_arr[0:10])

    # Vector of indices
    row_index = np.where(route_id_match_arr)

    return row_index

def main(*route_id_input):
    
    pass

    if len(route_id_input) == 0:
        show_routes = 1
    else:
        show_routes = 0
        route_id_input = route_id_input[0]

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
    trip_table = []
    with open(filename, newline='') as testCSV:
        reader = csv.DictReader(testCSV)
        #Loop through and read in data
        for row in reader:
            trip_table.append(row)

    #Loop through and assign dictionary keys to col_names
    col_names_view = trip_table[0].keys()
    col_names = []
    for key in col_names_view:
        col_names.append(key)



    #Loop through trip_table and get start_station, end_station and duration
    duration = []
    start_station = []
    end_station = []
    for row in trip_table:
        duration.append(row['duration'])
        start_station.append(row['start_station'])
        end_station.append(row['end_station'])

    # Convert lists into np arrays, and then to int type
    dur_arr = np.array(duration)
    d = dur_arr.astype(int)

    start_arr = np.array(start_station)
    s = start_arr.astype(int)

    end_arr = np.array(end_station)
    e = end_arr.astype(int)

    # vector of route identifiers (string concat of start+end)
    #route_id = np.transpose(np.transpose(ss)+np.transpose(ee))
    route_id_arr = [start_arr[i]+end_arr[i] for i in range(len(start_arr))]

    # Set of uniq identifiers
    route_id_set = set(route_id_arr)

    # Make Route ID lookup table by affixing indices to IDs (out: 2 x N_routes)
    route_table = []
    for index, route in enumerate(route_id_set):
        route_table.append((index, route))
        #if index == 5:
        #    print(route_table)

    #code.interact(local=locals())
    if show_routes:
        print(index, route in enumerate(route_id_set))
        return
    else:
        row_index = get_route_index(route_id_input, route_data=route_id_arr)

        durs = d[row_index]

        plt.hist(durs)
        plt.show()

    #code.interact(local=locals())
        return

if __name__ == '__main__':
    main()
