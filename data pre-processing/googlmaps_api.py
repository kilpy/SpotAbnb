import requests
import json
import pandas as pd
import math
from datetime import datetime
import googlemaps
from dotenv import load_dotenv
import os
import requests
import json
from typing import List, Dict, Any
import datetime
import numpy as np

class GoogleRoutesMatrix:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
        
    def compute_route_matrix(
        self, 
        origins: List[Dict[str, float]], 
        destinations: List[Dict[str, float]],
        travel_mode: str = "DRIVE",
        routing_preference: str = "TRAFFIC_AWARE",
        units: str = "METRIC"
    ) -> List[Dict]:
        
        formatted_origins = []
        for loc in origins:
            formatted_origins.append({
                "waypoint": {
                    "location": {
                        "latLng": {
                            "latitude": loc["latitude"],
                            "longitude": loc["longitude"]
                        }
                    }
                }
            })
        
        formatted_destinations = []
        for loc in destinations:
            formatted_destinations.append({
                "waypoint": {
                    "location": {
                        "latLng": {
                            "latitude": loc["latitude"],
                            "longitude": loc["longitude"]
                        }
                    }
                }
            })
        

        payload = {
            "origins": formatted_origins,
            "destinations": formatted_destinations,
            "travelMode": travel_mode,
            "routingPreference": routing_preference,
            "units": units
        }
        
        future_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)
        payload["departureTime"] = future_time.isoformat()
            
        # Headers required for the API
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "originIndex,destinationIndex,duration,distanceMeters,status"
        }
        
        response = requests.post(self.base_url, json=payload, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    def format_results(self, matrix_results: List[Dict]) -> List[Dict[str, Any]]:
       
        formatted_results = []
        
        # matrix_results is now a list, not a dictionary
        for route in matrix_results:
            duration_str = route.get("duration", "0s")
            duration_seconds = int(duration_str.rstrip("s")) if duration_str.endswith("s") else 0
            
            result = {
                "origin_index": route.get("originIndex"),
                "destination_index": route.get("destinationIndex"),
                "distance_meters": route.get("distanceMeters", 0),
                "distance_km": round(route.get("distanceMeters", 0) / 1000, 2),
                "duration_seconds": duration_seconds,
                "duration_minutes": round(duration_seconds / 60, 2),
                "status": "OK" if route.get("distanceMeters", 0) > 0 else "FAILED"
            }
            formatted_results.append(result)
            
        return formatted_results
    
def get_distances(event, airbnbs, GMAPS_API_KEY):
   
    # Initialize with API key
    api_key = GMAPS_API_KEY
    routes_matrix = GoogleRoutesMatrix(api_key)
    
    origins = [{"latitude": event['latitude'], "longitude": event['longitude']}]
    print(origins)
    destinations = airbnbs.to_dict(orient='records') # requires list of dicts with lat/long
    print(len(destinations))
    try:
        matrix_results = routes_matrix.compute_route_matrix(
            origins=origins,
            destinations=destinations,
            travel_mode="DRIVE",
            routing_preference="TRAFFIC_AWARE"
        )
        
        formatted_results = routes_matrix.format_results(matrix_results)
        for result in formatted_results:
            origin_idx = result["origin_index"]
            dest_idx = result["destination_index"]
            print(f"From {origins[origin_idx]} to {destinations[dest_idx]}:")
            print(f"  - Distance: {result['distance_km']} km")
            print(f"  - Duration: {result['duration_minutes']} minutes")
            print(f"  - Status: {result['status']}")
            print()
        return formatted_results
    except Exception as e:
        print(f"Error: {e}")

def calculate_distance_scores(event, df, GMAPS_API_KEY):
    ### temporary dataframe to store longitude, latitude of each centroid without overriding airbnb lat, long ###
    temp_df = df.copy()
    
    temp_df.sort_values(by='cluster_id', inplace=True)
    
    temp_df.reset_index(drop=True, inplace=True)

    temp_df.rename(columns={'centroid_longitude': 'longitude', 'centroid_latitude': 'latitude'}, inplace=True)
    
    temp_df = temp_df[['cluster_id', 'longitude', 'latitude']]

    temp_df = temp_df.drop_duplicates(subset=['cluster_id'])

    results = get_distances(event, temp_df, GMAPS_API_KEY)

    temp_df['distance_km'] = [x['distance_km'] for x in results]

    temp_df.sort_values(by='distance_km', inplace=True)
    temp_df.reset_index(drop=True, inplace=True)

    # assign scoring metrics to each airbnb based on distance from event
    temp_df['score'] = temp_df['distance_km'].apply(lambda x: 1/x ** (1/4) if x > 0 else 0)
    temp_df.sort_values(by='score', ascending=False, inplace=True)

    cluster_id_to_score = temp_df.set_index('cluster_id')['score'].to_dict()

    df['distance_score'] = df['cluster_id'].map(cluster_id_to_score)

    df.to_csv('final_airbnb_dataset.csv', index=False)

def calculate_scores(event, airbnb_df, GMAPS_API_KEY):
    
    calculate_distance_scores(event, airbnb_df, GMAPS_API_KEY)

    ## Price Scoring metric: ##
    airbnb_df['price_score'] = airbnb_df['price'].apply(score_price)

    airbnb_df['total_score'] = 0.5 * airbnb_df['distance_score'] + 0.5 * airbnb_df['price_score']

    airbnb_df['booked'] = airbnb_df['total_score'].apply(lambda x: 1 if x >= airbnb_df['total_score'].quantile(0.8) else 0)

    airbnb_df.to_csv('final_airbnb_dataset.csv', index=False)


def score_price(price, p25=147.84, median=183.36, p75=293):
    
    scale = (p75 - p25)  # Use IQR as scale parameter

    score = 1 / (1 + np.exp((price - median) / scale))
    
    return score


#df=pd.read_csv("events.csv")
#airbnb_df = pd.read_csv("final_airbnb_dataset.csv")
#load_dotenv()
#GMAPS_KEY = os.getenv("GMAPS_KEY")
#calculate_scores(df.iloc[0], airbnb_df, GMAPS_KEY) #in main to directly update final_airbnb_dataset.csv ###