"""A script to crawl posts from Bluesky for the January 2025 Southern California wildfires.
"""

# check if session is in Google Colab
try:
    import google.colab
    IN_COLAB = True
    print('Google Colab session!')
except:
    IN_COLAB = False
    print('Not a Google Colab session.')

# add src path to the notebook
import os
import sys
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT: str = '/content/drive/MyDrive/papers/2025b_relevance_2.0'
else:
    PROJECT_ROOT: str = os.path.dirname(os.path.abspath(os.path.dirname("__file__")))
if PROJECT_ROOT not in sys.path:
    sys.path.append(os.path.join(PROJECT_ROOT))
print(PROJECT_ROOT)

import json
import time
import pandas as pd
import src.bsky_search as bsky
from datetime import datetime, timedelta
from pathlib import Path
from tqdm.auto import tqdm

DATA_PATH: str = os.path.join(PROJECT_ROOT, 'data')
print(DATA_PATH)

socal_fire_keywords = [
    # Event types
    # redundancy actually makes sense here as we always get the top posts
    "fire", "wildfire", "firestorm", "brush fire", "forest fire", "blaze", "inferno",
    # Locations
    "southern california", "los angeles", "san diego", "socal",
    # Environmental conditions
    "santa ana", "drought", "low humidity", "red flag warning",
    # Impacts & response
    "evac", "fatality", "homes destroyed", "smoke plume", "air quality",
    # Hashtags
    "californiawildfires", "cafire", "socalfires", "sandiegofire", "santaanawinds",
]
print(socal_fire_keywords)

# Container for all parsed posts
all_parsed_posts = []

# Define the time window and interval
start_time = datetime(2024, 12, 24, 0, 0)  # 2 weeks before (24.12.2024)
end_time = datetime(2025, 2, 15, 0, 0)  # 2 weeks after (<15.2.2025)
interval = timedelta(minutes=30)  # 30-minute intervals

# Pre-compute all (since, until) intervals
intervals = []
current = start_time
while current < end_time:
    since = current.strftime("%Y-%m-%dT%H:%M:%SZ")
    until = (current + interval).strftime("%Y-%m-%dT%H:%M:%SZ")
    intervals.append((since, until))
    current += interval

# Build a task list of (keyword, since, until) for progress tracking
tasks = [
    (kw, since, until)
    for since, until in intervals
    for kw in socal_fire_keywords
]
print(f"Total tasks: {len(tasks)}")
print(tasks[:10])

all_parsed_posts: list = []
all_raw_posts: list = []

# Iterate with a tqdm progress bar
for kw, since, until in tqdm(tasks, desc="Fetching posts", unit="task"):
    parsed_results = []
    # Retry up to 10 times on failure
    for attempt in range(10):
        try:
            raw_results, parsed_results = bsky.fetch_all_posts(q=kw, since=since, until=until)
            break
        except Exception as e:
            if attempt < 9:
                time.sleep(5)  # brief pause before retry
                # print(e)
            else:
                print(f"Failed to fetch posts for '{kw}' between {since} and {until} after 10 attempts: {e}")
    all_parsed_posts.extend(parsed_results)
    all_raw_posts.extend(raw_results)
    time.sleep(1)

# Build a single DataFrame
bsky_socal_df: pd.DataFrame = pd.DataFrame(all_parsed_posts)
bsky_socal_raw = all_raw_posts

# Store all the results
Path(os.path.join(DATA_PATH, 'raw', '2025_socal_wildfires')).mkdir(parents=True, exist_ok=True)
with open(os.path.join(DATA_PATH, 'raw', '2025_socal_wildfires', 'bsky_socal_raw.json'), 'w') as final:
    json.dump(bsky_socal_raw, final)
bsky_socal_df.to_csv(os.path.join(DATA_PATH, 'raw', '2025_socal_wildfires', 'bsky_socal_df.csv'), index=False)

# Aand display
print(bsky_socal_df.head())