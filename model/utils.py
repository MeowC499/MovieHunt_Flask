import os
import gdown
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

def download_if_missing(file_path, gdrive_url):
    if not os.path.exists(file_path):
        print(f"📥 Downloading {file_path}")
        gdown.download(gdrive_url, file_path, quiet=False)

def load_data():
    os.makedirs("data", exist_ok=True)

    # Google Drive files
    gdrive_files = {
        "ratings.csv": "https://drive.google.com/uc?id=1Xuh5ZuI2RnBZ2f-nuKAkrI29dwaRzSXP",
        "movies.csv": "https://drive.google.com/uc?id=1P5p6bpyVA_uTGGeYq5PiK-MwIUZ5sIi1",
        "links.csv": "https://drive.google.com/uc?id=1AtRiQ0-X5KuZFjnPWcPj1kpqDe3KZUWq",
        "ncf_model_state.pth": "https://drive.google.com/uc?id=1-5JsDtm_EF3qwvTopoWDEUu17FVfYEaV"
    }

    for filename, url in gdrive_files.items():
        download_if_missing(f"data/{filename}", url)

    # Now load the data
    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")
    links = pd.read_csv("data/links.csv")

    # Label encoding
    user_encoder = LabelEncoder().fit(ratings["userId"])
    movie_encoder = LabelEncoder().fit(ratings["movieId"])
    user_to_index = dict(zip(ratings["userId"], user_encoder.transform(ratings["userId"])))
    movie_to_index = dict(zip(ratings["movieId"], movie_encoder.transform(ratings["movieId"])))

    return {
        "ratings": ratings,
        "all_movies": movies,
        "user_to_index": user_to_index,
        "movie_to_index": movie_to_index
    }
