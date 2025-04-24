import os
import json
import pandas as pd

# Step 1: Authenticate user and fetch Spotify data
print("Running Spotify API script...")
os.system("python3 spotify_api.py")

# Check if the Spotify data file was created
if not os.path.exists("spotify_data.json"):
    raise FileNotFoundError("spotify_data.json not found. Please check spotify_api.py execution.")

print("Spotify data fetched successfully.")
#verified till here

# Step 2: Get concert and Airbnb data using Ticketmaster and Google Maps API
print("Running Ticketmaster and Google Maps API script...")
os.system("python3 ticketmaster_api.py")

# Check if the concert data file was created
if not os.path.exists("events.csv"):
    raise FileNotFoundError("events.csv not found. Please check ticketmaster_api.py execution.")

# Check if the updated Airbnb dataset was created
if not os.path.exists("final_airbnb_dataset.csv"):
    raise FileNotFoundError("final_airbnb_dataset.csv not found. Please check ticketmaster_api.py execution.")

print("Events and updated Airbnb dataset generated successfully.")