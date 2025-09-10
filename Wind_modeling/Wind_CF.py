import requests
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
from math import exp
from windpowerlib import WindTurbine, ModelChain   # there is a python library for wind, similar to PVlibe (Gamesa turbine not available there though)


# 1. Define location, year and Turbine specification
api_key = 'gfHs8e7Vd44QfrTw96pNjTecBeGrQ98x7JgChlZZ'  # Replace this. API same as for PV modeling
email = 'ac11992@nyu.edu'# Replace this
full_name = 'Alexandre Cattry'# Replace this
affiliation = 'NYU'
reason = 'research'

lat = 38.95
lon = -90.56
year = 2009

rated_power_kw = 2500 # [kW] Gamesa G126/2500  
turbine_loss=0.19 # [-] turbine efficiency loss  (19% loss)
cut_out_speed=21 # [m/s]  cut out speed Gamesa G126/2500  
reentry_margin = 5  # m/s Hysteresis effect 
Temperature_cutoff=243.15 # [K] -30 degree C

# 2. Download data
# --- Construct WKT (Well-Known Text) string
wkt = f"POINT({lon} {lat})"

# --- WIND Toolkit API endpoint
url = 'https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-download.csv'

# --- Attributes at 100 m
attributes = [
    'windspeed_100m',
    'relativehumidity_2m',
    'temperature_100m',
    'pressure_100m',
]

# --- Request parameters
params = {
    'api_key': api_key,
    'wkt': wkt,
    'names': str(year),
    'interval': '60', #hourly data
    'utc': 'true',
    'leap_day': 'false',
    'email': email,
    'full_name': full_name,
    'affiliation': affiliation,
    'reason': reason,
    'attributes': ','.join(attributes)
}

# --- Make the request
print("Sending request...")
response = requests.get(url, params=params)

if response.status_code != 200:
    print("❌ Error downloading data:")
    print(response.text)
else:
    print("✅ Download successful, saving to CSV...")

    # Save raw CSV file
    filename = f'wind_100m_{lat}_{lon}_{year}.csv'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"📁 Saved to {filename}")


# --- 3. Load CSV and preprocess
df = pd.read_csv(filename, skiprows=1)

# Rename columns for convenience
df.columns = [c.strip() for c in df.columns]
df.rename(columns={
    'wind speed at 100m (m/s)': 'v_meas',
    'relative humidity at 2m (%)': 'rh',
    'air temperature at 100m (C)': 'T_C',
    'air pressure at 100m (Pa)': 'p_meas'
}, inplace=True)

# --- 4. Calculate air density (n_meas)
T_K = df['T_C'] + 273.15 # convert Celsius to Kelvin
phi = df['rh'] / 100.0 
p_meas = df['p_meas']

# Tetens equation for vapor pressure
p_water = phi * 610.78 * np.exp((17.27 * df['T_C']) / (df['T_C'] + 237.3))

R_dry = 287.1 # [J/(kg K)]
R_water = 461.5 # [J/(kg K)]

n_meas = (p_meas - p_water) / (R_dry * T_K) + p_water / (R_water * T_K)

# --- 5. Correct wind speed to standard air density
v_meas = df['v_meas']
n_std = 1.225
v_std = v_meas * (n_meas / n_std) ** (1 / 3)

# --- 6.  Capacity factor values at given wind speeds
df_cf = pd.read_csv("gamesa_g126_cf_curve.csv")

# Extract the columns into NumPy arrays
wind_speeds_cf = df_cf["wind_speed_m_per_s"].values
cf_values = df_cf["capacity_factor"].values


# Convert CF to actual power output (kW)
power_output_cf = rated_power_kw * cf_values

# Interpolate turbine power using CF curve
turbine_power = np.interp(v_std, wind_speeds_cf, power_output_cf)


# --- 6b. Apply loss mechanisms to turbine power

# 1. Uniform 19% loss
turbine_power_loss = turbine_power * (1-turbine_loss)

# 2. Temperature cutoff: T_K <= 243.15 K
cold_condition_mask = T_K <=  Temperature_cutoff
turbine_power_loss[cold_condition_mask] = 0

# 3. Cut-out hysteresis logic
reentry_speed = cut_out_speed - reentry_margin

# Identify hours with windspeed above cut-out
shutdown = np.zeros_like(v_std, dtype=bool)
in_shutdown = False
for i in range(len(v_std)):
    if in_shutdown:
        if v_std[i] < reentry_speed:
            in_shutdown = False
    elif v_std[i] > cut_out_speed:
        in_shutdown = True
    shutdown[i] = in_shutdown

# Apply hysteresis shutdown
turbine_power_loss[shutdown] = 0

# Add final adjusted power to DataFrame
df['turbine_power_kw'] = turbine_power_loss


# --- 7. Add to DataFrame and save
df['v_std'] = v_std

processed_filename = f'wind_power_output_{lat}_{lon}_{year}.csv'
df.to_csv(processed_filename, index=False)
print(f"📁 Processed results saved to {processed_filename}")

# Group by 'Hour' and take the mean of turbine power
average_hourly_power = df.groupby('Hour')['turbine_power_kw'].mean()/rated_power_kw

# 8. Plot and save the figure
plt.figure(figsize=(10, 5))
average_hourly_power.plot(kind='line', color='skyblue', marker='o')
plt.title('Average Turbine CF Output by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average CF (-)')
plt.grid(True)
plt.tight_layout()

# Compute total energy
total_energy_kwh = df['turbine_power_kw'].sum()
energy_text = f"⚡ Total energy: {total_energy_kwh:,.0f} kWh"

# Add as text to the figure (lower right corner)
plt.text(0.99, 0.01, energy_text,
         transform=plt.gca().transAxes,
         fontsize=10, va='bottom', ha='right',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

# Save the figure with full context in filename
plot_filename = f'average_hourly_turbine_CF_{lat}_{lon}_{year}.png'
plt.savefig(plot_filename, dpi=300)
print(f"📁 Figure saved as '{plot_filename}'")



# Create a datetime column if not already present
df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
df['Month'] = df['timestamp'].dt.month

# Energy per hour in kWh already, sum per month and convert to MWh
monthly_energy_mwh = df.groupby('Month')['turbine_power_kw'].sum() / 1000

# Plot monthly energy
plt.figure(figsize=(10, 5))
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.bar(months, monthly_energy_mwh, color='mediumseagreen')
plt.title(f'Total Monthly Energy Generation ({year})')
plt.xlabel('Month')
plt.ylabel('Energy [MWh]')
plt.grid(axis='y')
plt.tight_layout()

# Save the figure
monthly_plot_filename = f'monthly_energy_bar_plot_{lat}_{lon}_{year}.png'
plt.savefig(monthly_plot_filename, dpi=300)
print(f"📁 Monthly bar plot saved as '{monthly_plot_filename}'")



