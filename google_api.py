import googlemaps
import datetime
import simplejson
import numpy as np
import code

gmaps = googlemaps.Client(key='AIzaSyA-Lvz3fLHg0Uk1V3L3p2mE4mkTIoozSEo')

dummy_date = datetime.date.today() + datetime.timedelta(days=1)
time_test1 = datetime.time(0,4)
time_test2 = datetime.datetime.combine(dummy_date,time_test1)

print(time_test2)
print(type(time_test2))


start_lat = 39.956619
start_long = -75.198624
start_loc = (start_lat,start_long)

end_lat = 39.949741
end_long = -75.180969
end_loc = (end_lat,end_long)

#print(start_loc)
#print(end_loc)

dist_result = gmaps.distance_matrix(start_loc,end_loc, mode = 'bicycling', language = 'english', departure_time = time_test2)



#print simplejson.dumps([s['formatted_address'] for s in result['results']], indent=2)
#print(dist_result)
#print(dist_result['rows'])

#print(dist_result['rows'][0])
#print(dist_result['rows'][0].keys())

#print(dist_result['rows'][0]['elements'])

#print(dist_result['rows'][0]['elements'][0].keys())

#print(dist_result['rows'][0]['elements'][0]['duration'])
#print(dist_result['rows'][0]['elements'][0]['duration'].keys())

#print(dist_result['rows'][0]['elements'][0]['duration']['value'])

#print(type(dist_result['rows'][0]['elements'][0]['duration']['value']))


 # We have only one row (origin) but several destinations (columns)
duration_est = []
for thing in dist_result['rows'][0]['elements']:
    duration_est.append(thing['duration']['value'])
    duration_est = np.array(duration_est)

print(duration_est)
print(type(duration_est[0]))


code.interact(local=locals())


#print(type(dist_result))
#print(type(dist_result["rows"]))


# Geocoding an address
#geocode_result = gmaps.geocode('1600 Amphitheatre Parkway, Mountain View, CA')
