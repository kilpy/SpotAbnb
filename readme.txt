# AUTHORS: Arzoo Jiwani, Brent Garey

# GITHUB: https://github.com/kilpy/SpotAbnb

### REQUIRED FILES: ###
ensure you have a file called .env that contains the following API keys:
DISCOVERY_KEY = 3fw2o6oLbQrHGokCdGJmxUGxHlgCYhUD
GMAPS_KEY = AIzaSyBI7YGg2J1Ky_KRWtsGW_JG0t6UyenPzRc
SPOTIFY_KEY = KraVYTRQzL31I7TleeO3d8AwQyMkaWwA
########################

HOW TO RUN:

Open main.py and run the file. 

IMPORTANT: on line 19, the function call to spotify's api is commented out. 
The way Spotify manages their tokens is that they expire every hour due to security. 
As a work around, I've left my spotify_data.json file within.
Simply running main.py will take the data extracted from this file for the rest of the file (which should be fine). 

HOWEVER, should you want to run it with your own spotify account login, then please email us. (jiwani.a@northeastern.edu or garey.b@northeastern.edu) 
Spotify's API requires us to approve users with a token that expires. We will try to add you before you test it, but we need your email associated with spotify to do so. 

Once added, AND uncommented line 19, delete spotify_data.json (and .cache if it is there). 
This should force you to log in to spotify.
##############################
Overview

1. Authenticates and fetches user data from Spotify.

2. Recommends nearby concerts using the Ticketmaster API.

3. Finds Airbnbs near the concerts.

4. Scores Airbnbs based on distance, price, and Google Maps routing data.

5. Ranks recommendations using XGBoost and Deep Neural Networks.

6. Opens a browser-based report with top suggestions.
###############################
Technologies & APIs Used

Spotify API – To get user’s top artists and genres.

Ticketmaster API – To find concert events based on Spotify data.

Google Maps API – For distance and travel time calculations.

XGBoost – For machine learning-based Airbnb recommendations.

Keras/TensorFlow (DNN) – For deep learning recommendations.

Pandas – For data manipulation.

dotenv – To manage API keys securely.

#####################################