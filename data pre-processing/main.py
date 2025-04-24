import os
import json
import pandas as pd
from dotenv import load_dotenv
from googlmaps_api import calculate_scores 
import webbrowser
from dnn import run_dnn_model

# Step 1: Authenticate user and fetch Spotify data
print("🎵 Running Spotify API script...")
os.system("python3 spotify_api.py")

# Check if the Spotify data file was created
if not os.path.exists("spotify_data.json"):
    raise FileNotFoundError("spotify_data.json not found. Please check spotify_api.py execution.")

print("Spotify data fetched successfully.")

# Step 2: Get concert and Airbnb data using Ticketmaster and Google Maps API
print("🎤 Running Ticketmaster and initial Google Maps script...")
os.system("python3 ticketmaster_api.py")

# Check if the concert data file was created
if not os.path.exists("events.csv"):
    raise FileNotFoundError("events.csv not found. Please check ticketmaster_api.py execution.")

# Check if the updated Airbnb dataset was created
if not os.path.exists("final_airbnb_dataset.csv"):
    raise FileNotFoundError("final_airbnb_dataset.csv not found. Please check ticketmaster_api.py execution.")

print("Events and Airbnb dataset updated.")

# Step 3: Google Maps advanced scoring (distance + price)
print("Running advanced scoring using Google Maps Routes Matrix...")

# Load data
df = pd.read_csv("events.csv")
airbnb_df = pd.read_csv("final_airbnb_dataset.csv")

# Load environment variables
load_dotenv()
GMAPS_KEY = os.getenv("GMAPS_KEY")

# Calculate scores and update Airbnb dataset
calculate_scores(df.iloc[0], airbnb_df, GMAPS_KEY)

print("Scoring complete. final_airbnb_dataset.csv updated with distance, price, and total score.")

print("Step 4: Running DNN model and generating recommendations...")
os.system("python3 dnn.py")

# Step 5: Open the HTML report with top Airbnb links

print("\nRunning DNN model for Airbnb prediction...")
event_name = df.iloc[0]['event_name'] if 'event_name' in df.columns else "an upcoming concert"
dnn_html = run_dnn_model(event_name)

print("🌍 Opening DNN Report in browser...")
webbrowser.open(f"file://{dnn_html}")