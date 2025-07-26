import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# Get environment variables
client_id = os.getenv('SPOTIPY_CLIENT_ID')
client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
redirect_uri = os.getenv('SPOTIPY_REDIRECT_URI')


# Setup auth
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope="user-library-read playlist-read-private"
))

# --- 1. Load or simulate data ---
# Replace this with your actual Spotify data (from the API)
data = pd.DataFrame({
    'danceability': np.random.rand(100),
    'energy': np.random.rand(100),
    'valence': np.random.rand(100),
    'tempo': np.random.normal(120, 10, 100),
    'acousticness': np.random.rand(100),
    'popularity': np.random.randint(20, 90, 100),
    'song_name': [f"Song {i}" for i in range(100)]
})

# --- 2. Normalize features ---
features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'popularity']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# --- 3. Run K-Means clustering ---
k = 4  # You can experiment with different values
kmeans = KMeans(n_clusters=k, random_state=42)
data['cluster'] = kmeans.fit_predict(data_scaled)

# --- 4. Visualize clusters ---
sns.pairplot(data, hue='cluster', vars=features[:4])
plt.suptitle("Clustered Songs Based on Audio Features", y=1.02)
plt.show()

# --- 5. Analyze clusters ---
cluster_summary = data.groupby('cluster')[features].mean()
print("\nCluster Feature Averages:")
print(cluster_summary)

# Optional: Show top 3 songs per cluster
for c in range(k):
    print(f"\nCluster {c} sample songs:")
    print(data[data['cluster'] == c]['song_name'].head(3).to_string(index=False))


def main():
    # your main code here

    if __name__ == "__main__":
        main()