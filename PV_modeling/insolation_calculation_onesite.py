import string
import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem, Array, FixedMount
from pvlib.modelchain import ModelChain
from pvlib.iotools import get_psm3
import matplotlib.pyplot as plt

# 1. Define location and year
lat, lon = 34.05, -118.25   # Example: Los Angeles, CA

year = 2022
api_key = 'XXXXXXXX'  # Replace with your actual API key
email = 'XXXXXX@nyu.edu'  # Replace with your actual email
# 2. Download NSRDB data
data, meta = get_psm3(latitude=lat, longitude=lon, names=str(year), api_key=api_key, email=email, map_variables=True)

# 3. Localize to site's timezone
data.index = data.index.tz_convert('America/Los_Angeles')

# Export the NSRDB data to a CSV file
data.to_csv('outputs/nsrdb_data_' + str(year) + '.csv')
