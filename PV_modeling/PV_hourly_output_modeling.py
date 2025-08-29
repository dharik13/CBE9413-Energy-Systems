import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem, Array, FixedMount, SingleAxisTrackerMount
from pvlib.modelchain import ModelChain
from pvlib.iotools import get_psm3

# 1. Define location and year
lat, lon = 34.05, -118.25   # Example: Los Angeles, CA
# lat, lon = 42.3555, -71.0565  # Example: Boston, MA

year = 2022
api_key = 'XXXXXXXXXX'  # Replace with your actual API key
email = 'XXXXX@XXXX.edu'  # Replace with your actual email
# 2. Download NSRDB data for the particular location
data, meta = get_psm3(latitude=lat, longitude=lon, names=str(year), api_key=api_key, email=email, map_variables=True)

# 3. Localize to site's timezone
data.index = data.index.tz_convert('America/Los_Angeles')


location = Location(latitude=lat, longitude=lon, tz='America/Los_Angeles', altitude=meta['altitude'])
# 4. Define PV system parameters
# To understand further- refer to 
# 1 ) https://pvlib-python.readthedocs.io/en/stable/user_guide/modeling_topics/pvsystem.html#
# 2) Brown, P. R., & O'Sullivan, F. M. (2020). Spatial and temporal variation in the value of solar power across United States electricity markets. 
# Renewable and Sustainable Energy Reviews, 121, 109594.
PlantCapacityDC = 100 # System capacity (MW) of PV system in desired units

# System specific parameters
surface_tilt = lat  # Surface tilt angle (degrees), can be set to latitude for fixed tilt systems
azimuth_angle = 180.0  # Azimuth angle (degrees), 180 for south-facing
module_power = 400  # Power rating of the module (W)
modules_per_string = 1000  # Number of modules in series per string
strings_per_inverter = 10  # Number of strings for each inverter
dc_ac_ratio = 1.2  # DC to AC ratio, typically between 1.1 and 1.5
inverter_nom_loss = 0.04  # Inverter nominal loss (4%)
module_params = {'pdc0': module_power, 'gamma_pdc': -0.004}
SystemCapacityDC = module_power * modules_per_string * strings_per_inverter # DC capacity handled by each inverter (W)
inverter_params = {
        'pdc0': SystemCapacityDC,
        'pdc': SystemCapacityDC / dc_ac_ratio,
        'eta_inv_nominal': 1 - inverter_nom_loss
    }

# system losses parameters (percent units)
# Source:  A. P. Dobos, "PVWatts Version 5 Manual"  http://pvwatts.nrel.gov/downloads/pvwattsv5.pdf
losses = {
        'soiling': 2,
        'shading': 3,
        'snow': 0,
        'mismatch': 2,
        'wiring': 2,
        'connection': 0.5,
        'lid': 1,
        'nameplate_rating': 0.0,
        'age': 0.0,
        'availability': 3
    }


#5.  Define PV system
# Specifying type of mounting - fixed tilt or single axis tracking
# NOTE: COMMENT ONE OF THE MOUNT DEFINITION FUNCTIONS
## Fixed tilt system definition:
# mount = FixedMount(surface_tilt=surface_tilt,
#                     surface_azimuth=180)
## Single axis tracking system definition:
mount = SingleAxisTrackerMount(
        axis_tilt=surface_tilt,
        axis_azimuth=azimuth_angle,   # Assuming south-facing axis
        backtrack=True,     # Enable backtracking for the tracker
        gcr=0.33,  # Ground coverage ratio
        max_angle=60.0,      # Maximum angle for backtracking
)

# Define the PV array function using the specified mount
array = Array(
    mount=mount,
    module='pvwatts_dc',
    module_parameters=module_params, 
    modules_per_string=modules_per_string,
    strings= strings_per_inverter, # Number of strings in parallel part of the system
    temperature_model_parameters=pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
)

# Define the PV system, including inverter
# The PVSystem represents one inverter and the PV modules that supply DC power to the inverter.
system = PVSystem(arrays=[array], 
                  inverter_parameters=inverter_params,
                  module='pvwatts_ac',
                  modules_per_string=modules_per_string,
                  strings_per_inverter=strings_per_inverter,
                losses_parameters=losses,
                  albedo='urban')
mc = ModelChain(system, location, 
                aoi_model='physical', 
                spectral_model='no_loss',
                dc_model='pvwatts', 
                ac_model='pvwatts',
                temperature_model='sapm',
                transposition_model='perez',
                losses_model='pvwatts')

# 6. Run the model to simulate system performance
mc.run_model(weather=data)

# 7. Results at the plant-level
NumberofSystems = PlantCapacityDC / (SystemCapacityDC/1e+6)  # Number of systems in parallel needed to meet the plant capacity
ac_output = mc.results.ac/1e+6 * NumberofSystems  # Hourly AC power output (MW)
dc_output = mc.results.dc/1e+6 * NumberofSystems  # Hourly DC power output (MW)

# Annual capacity factor calculation
annual_ac_CF = ac_output.sum()/(PlantCapacityDC/dc_ac_ratio*8760)
annual_dc_CF = dc_output.sum()/(PlantCapacityDC*8760)

# Combine AC and DC outputs into a single DataFrame and save to CSV
output_df = pd.DataFrame({
    'ac_output': ac_output,
    'dc_output': dc_output
})
print("SystemCapacityDC:", SystemCapacityDC/1e+6)
print("Number of Systems:", NumberofSystems)
print("max(ac_output):", max(ac_output))
print("max(dc_output):", max(dc_output))
print("AC Capacity Factor:", annual_ac_CF)
print("DC Capacity Factor:", annual_dc_CF)
output_df.to_csv('pv_output_hourly.csv')