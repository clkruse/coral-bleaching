from tqdm import trange
from datetime import date, datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import ee
import json
import pandas as pd
import csv
plt.rcParams['figure.figsize'] = [10, 9]

def read_db(db_name):
    events = []
    database = csv.DictReader(open(db_name))
    for row in database:
        # Check if any elements are missing. If so, ignore that record
        if all(value != '' for value in row.values()):
            event = {
                "lat": float(row['Lat']),
                "lng": float(row['Lng']),
                "year": int(row['Year']),
                "month": int(row['Month']),
                "severity": int(row['Severity'])
            }
            events.append(event)
    return(events)

def get_time(db_entry, window):
    year = db_entry['year']
    month = db_entry['month']
    event_date = date(year, month, 1)
    unix_time = time.mktime(event_date.timetuple())
    start_time = unix_time - 3600 * 24 * window
    start_date = str(date.fromtimestamp(start_time))
    end_time = unix_time + 3600 * 24 * window
    end_date = str(date.fromtimestamp(end_time))
    start_date = start_date
    end_date = end_date
    return(start_date, end_date)

def get_time_series(product, db_entry, window):
    img_col = ee.ImageCollection(products[product]['url'])\
        .select(products[product]['band'])\
        .filterDate(*get_time(db_entry, window))
    buffer_radius = 100000

    location = ee.Geometry.Point(db_entry['lng'], db_entry['lat']-.5).buffer(buffer_radius)

    try:
        info = img_col.getRegion(location, buffer_radius*2).getInfo()
    except:
        return([], [])
    header = info[0]
    data = np.array(info[1:])

    try:
        time_index = header.index('time')
        times = [datetime.fromtimestamp(i/1000) for i in (data[0:,time_index].astype(int))]

        values_index = header.index(products[product]['band'])
        values = data[0:, values_index].astype(np.float)
        values *= products[product]['scale']
        values = values[~np.isnan(values)].tolist()
    except:
        return([], [])
    return(times, values)


products = {
    "sst": {
        "url": "NOAA/CDR/OISST/V2",
        "band": "sst",
        "scale": 0.01
        },
    #Note: WHOI model stops fairly short of coastlines
    "sst_whoi": {
        "url": "NOAA/CDR/SST_WHOI/V2",
        "band": "sea_surface_temperature",
        "scale": 1
        },
    "sst_avhrr": {
        "url": "NOAA/CDR/SST_PATHFINDER/V53",
        "band": "sea_surface_temperature",
        "scale": 0.01
        },
    "chlor_a_terra": {
        "url": "NASA/OCEANDATA/MODIS-Terra/L3SMI",
        "band": "chlor_a",
        "scale": 1
        },
    "chlor_a_aqua": {
        "url": "NASA/OCEANDATA/MODIS-Aqua/L3SMI",
        "band": "chlor_a",
        "scale": 1
        },
    "chlor_a_seawifs": {
        "url": "NASA/OCEANDATA/SeaWiFS/L3SMI",
        "band": "chlor_a",
        "scale": 1
        },
    "salinity": {
        "url": "HYCOM/sea_temp_salinity",
        "band": "salinity_0",
        "scale": 0.001
        },
    "velocity_u": {
        "url": "HYCOM/sea_water_velocity",
        "band": "velocity_u_0",
        "scale": 0.001
        },
    "velocity_v": {
        "url": "HYCOM/sea_water_velocity",
        "band": "velocity_v_0",
        "scale": 0.001
        },
    "cloud_transmission": {
        "url": "NOAA/CDR/PATMOSX/V53",
        "band": "cloud_transmission_0_65um",
        "scale": 0.00393701
        },
    "wind_avhrr": {
        "url": "NOAA/CDR/SST_PATHFINDER/V53",
        "band": "wind_speed",
        "scale": 1
        },
}


ee.Initialize()
database = read_db('bleaching.csv')
window_size = 10
output_fn = 'raw_window_10_params_11.json'

num_samples = len(database)
#num_samples = 100

for i in trange(num_samples):
    for product in products:
        t, v = get_time_series(product, database[i], window_size)
        database[i]['raw_' + product] = v
    if i % 50 == 0:
        with open(str(i) + "_" + output_fn, 'w') as fn:
            json.dump(database, fn, indent=4)
with open("complete_" + output_fn, 'w') as fn:
    json.dump(database, fn, indent=4)
