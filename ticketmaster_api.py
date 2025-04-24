import requests
import json
import pandas as pd

API_KEY = 'KraVYTRQzL31I7TleeO3d8AwQyMkaWwA'
def read_spotify():
    try: 
        with open("spotify_data.json","r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: Sotify data file not found.")

def get_concerts(artists, genre, city="New York"):
    url = f"https://app.ticketmaster.com/discovery/v2/events.json"
    search_terms = artists + [genre] if genre else artists

    # ✅ Initialize the dataframe outside the loop
    events_df = pd.DataFrame(columns=['event_name', 'event_date', 'venue_name', 'venue_location', 'longitude', 'latitude'])

    for term in search_terms:
        params = {
            'apikey': API_KEY,
            'keyword': term,
            'city': city,
            'classificationName': 'music',
            'size': 10
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            events = data.get('_embedded', {}).get('events', [])
            if events:
                print(f"\nConcerts in {city} for {term}:")
                for event in events:
                    event_name = event['name']
                    event_date = event['dates']['start']['localDate']
                    venue_name = event['_embedded']['venues'][0]['city']['name']
                    venue_location = event['_embedded']['venues'][0]['city']['name']
                    longitude = event['_embedded']['venues'][0]['location']['longitude']
                    latitude = event['_embedded']['venues'][0]['location']['latitude']
                    print(f"\n{event_name} \n{event_date} at {venue_name}, {venue_location}")
                    print(f"Longitude: {longitude}, Latitude: {latitude}")

                    # ✅ Add to dataframe
                    events_df.loc[len(events_df)] = [event_name, event_date, venue_name, venue_location, longitude, latitude]
        else:
            print(f"No concerts found for: {term} (Status: {response.status_code})")

    # ✅ Save after the loop ends
    if not events_df.empty:
        events_df.to_csv('events.csv', index=False)
        print("✅ Saved events to events.csv")
    else:
        print("⚠️ No events found for any artist/genre.")


# spotify_data = read_spotify()
# if spotify_data:
#     get_concerts(spotify_data["top_artists"], spotify_data["top_genre"])

