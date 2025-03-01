import requests
import pandas as pd
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

OPENCAGE_API_KEY = os.getenv("OPENCAGE_API_KEY")

if not OPENCAGE_API_KEY:
    print("Error: OPENCAGE_API_KEY is not set in the .env file.")
    exit()

location_name = input("Enter the location (City, Country): ")

GEOCODE_URL = f"https://api.opencagedata.com/geocode/v1/json?q={location_name}&key={OPENCAGE_API_KEY}"
geo_response = requests.get(GEOCODE_URL).json()

if "results" not in geo_response or len(geo_response["results"]) == 0:
    print("Location not found. Full API response:", geo_response)
    exit()

LATITUDE = geo_response["results"][0]["geometry"]["lat"]
LONGITUDE = geo_response["results"][0]["geometry"]["lng"]

print(f"Latitude: {LATITUDE}, Longitude: {LONGITUDE}")  

START_DATE = "20220101"
END_DATE = "20241231"

NASA_API_URL = (
    f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
    f"parameters=T2M,WS10M,ALLSKY_SFC_SW_DWN,PS"
    f"&community=RE&longitude={LONGITUDE}&latitude={LATITUDE}"
    f"&start={START_DATE}&end={END_DATE}&format=JSON"
)

response = requests.get(NASA_API_URL)
data = response.json()

timestamps = list(data["properties"]["parameter"]["T2M"].keys())
temperatures = list(data["properties"]["parameter"]["T2M"].values())
wind_speeds = list(data["properties"]["parameter"]["WS10M"].values())
solar_radiation = list(data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"].values())
air_pressure = list(data["properties"]["parameter"]["PS"].values())


df = pd.DataFrame({
    "timestamp": timestamps,
    "temperature_C": temperatures,
    "wind_speed_mps": wind_speeds,
    "solar_radiation_Wm2": solar_radiation,
    "air_pressure_Pa":air_pressure
})

df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H")

df.to_csv("weather_data.csv", index=False)

print(f"Weather data for {location_name} saved successfully!")
