

import spotipy
from collections import Counter
from spotipy.oauth2 import SpotifyOAuth
import json


# Spotify API Credentials
CLIENT_ID = "37c76da637124a1397a88686bc1b8278"
CLIENT_SECRET = "37c2f9cd367745a1800216e5e166822f"
REDIRECT_URI = "http://127.0.0.1:8888/callback"  
SCOPE = "user-top-read"

def get_top_artist_genre():
    sp_oauth = SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI, scope=SCOPE)
    token_info = sp_oauth.get_cached_token()
    
    if not token_info:
        print("Fetching new access token...")
        token_info = sp_oauth.get_access_token(as_dict=True)

    if token_info:
        access_token = token_info["access_token"]
        sp = spotipy.Spotify(auth=access_token)

        top_artists = sp.current_user_top_artists(limit=5)

        if "items" in top_artists and len(top_artists["items"]) > 0:
            all_genres = []
            artist_names = []
            for artist in top_artists["items"]:
                all_genres.extend(artist["genres"])
                artist_names.append(artist["name"])

            genre_counts = Counter(all_genres)
            favorite_genre = genre_counts.most_common(1)[0][0] if genre_counts else "Unknown"

            spotify_data = {
                "top_artists": artist_names,
                "top_genre": favorite_genre
            }
            with open("spotify_data.json", "w") as f:
                json.dump(spotify_data, f)

            print("\nYour top artists:")
            for idx, artist in enumerate(artist_names, start=1):
                print(f"{idx}. {artist}")

            print(f"\nYour Top Genre: {favorite_genre}")
            print("\nSaved data to spotify_data.json")
        else:
            print("No top artist data found. Try playing more music on Spotify.")
    else:
        print("Error: Could not retrieve access token.")

get_top_artist_genre()