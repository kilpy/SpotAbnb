import os
import json
import pandas as pd
from dotenv import load_dotenv
from googlemaps_api import calculate_scores 
import webbrowser
from dnn import run_dnn_model
import googlemaps_api
import dnn
import spotify_api
import ticketmaster_api
import XGBoost

# Step 1: Authenticate user and fetch Spotify data
print("🎵 Running Spotify API script...")

# Uncomment below to run the Spotify API script:
    # Requires user authentication from the authors (Arzoo and Brent) that once you are logged in, you can run the script without needing to re-authenticate.
# spotify_api.get_top_artist_genre()

# Check if the Spotify data file was created
if not os.path.exists("spotify_data.json"):
    raise FileNotFoundError("spotify_data.json not found. Please check spotify_api.py execution.")

print("Spotify data fetched successfully.")

# Step 2: Get concert and Airbnb data using Ticketmaster and Google Maps API
print("🎤 Running Ticketmaster and initial Google Maps script...")

spotify_data = ticketmaster_api.read_spotify()
if spotify_data:
    ticketmaster_api.get_concerts(spotify_data["top_artists"], spotify_data["top_genre"])


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
googlemaps_api.calculate_scores(df.iloc[0], airbnb_df, GMAPS_KEY)

print("Scoring complete. final_airbnb_dataset.csv updated with distance, price, and total score.")

# XGBoost model
print("Step 4: Running XGBoost model and generating recommendations...")
XGBoost.run_xgboost_model(df.iloc[0]['event_name'] if 'event_name' in df.columns else "an upcoming concert")

print("Step 5: Running DNN model and generating recommendations...")

# Step 5: Open the HTML report with top Airbnb links

print("\nRunning DNN model for Airbnb prediction...")
event_name = df.iloc[0]['event_name'] if 'event_name' in df.columns else "an upcoming concert"
dnn_html = run_dnn_model(event_name)

print("🌍 Opening DNN Report in browser...")
webbrowser.open(f"file://{dnn_html}")