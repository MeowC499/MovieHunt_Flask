from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import torch
import torch.nn as nn
import os
import logging
import requests

from model.ncf_model import NCF

app = Flask(__name__)

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "app.log")),
        logging.StreamHandler()
    ]
)

# GDrive file IDs
MODEL_PATH = "data/ncf_model_state.pth"
MODEL_ID = "1-5JsDtm_EF3qwvTopoWDEUu17FVfYEaV"
CSV_FILES = {
    "data/movies.csv": "1P5p6bpyVA_uTGGeYq5PiK-MwIUZ5sIi1",
    "data/ratings.csv": "1Xuh5ZuI2RnBZ2f-nuKAkrI29dwaRzSXP",
    "data/links.csv": "1AtRiQ0-X5KuZFjnPWcPj1kpqDe3KZUWq",
    "data/genome-tags.csv": "1ijNcsOK2b0Yenl2q8AgGYJwqLAbbtBEb",
    "data/genome-scores.csv": "1VTdzDeknOLCuO6CQq0QyJYjZvFj8SG7r",
    "data/tags.csv": "1F56JAP1jC5Ia1-3eB1tJajpFOiuY0Gfb"
}

data = None
model = None

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def download_file_from_gdrive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    logging.info(f"Downloaded: {destination}")

def download_data():
    for path, file_id in CSV_FILES.items():
        if not os.path.exists(path):
            logging.info(f"Downloading data file: {path}")
            download_file_from_gdrive(file_id, path)

def download_model():
    if not os.path.exists(MODEL_PATH):
        logging.info("Downloading model...")
        download_file_from_gdrive(MODEL_ID, MODEL_PATH)

def load_data():
    try:
        all_movies = pd.read_csv("data/movies.csv")
        if os.path.exists("data/user_feedback.csv"):
            user_feedback = pd.read_csv("data/user_feedback.csv")
        else:
            user_feedback = pd.DataFrame(columns=["username", "favorites", "feedback"])

        unique_users = user_feedback["username"].dropna().unique().tolist()
        if not unique_users:
            unique_users = ["default_user"]

        user_to_index = {user: idx for idx, user in enumerate(unique_users)}
        movie_to_index = {row["movieId"]: idx for idx, row in all_movies.iterrows()}

        logging.info(f"Loaded {len(all_movies)} movies and {len(user_to_index)} users.")
        return {
            "all_movies": all_movies,
            "user_to_index": user_to_index,
            "movie_to_index": movie_to_index
        }
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return {
            "all_movies": pd.DataFrame(),
            "user_to_index": {"fallback": 0},
            "movie_to_index": {}
        }

def initialize_app():
    global data, model
    if data is None or model is None:
        logging.info("Initializing app data and model...")
        download_data()
        download_model()
        data = load_data()

        model = NCF(
            num_users=len(data["user_to_index"]) or 1,
            num_movies=len(data["movie_to_index"]) or 1,
            latent_dim=100,
            hidden_dims=[256, 128, 64],
            dropout_rate=0.3,
            use_batchnorm=True
        )

        if os.path.exists(MODEL_PATH):
            try:
                model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
                model.eval()
                logging.info("Model loaded successfully.")
            except Exception as e:
                logging.warning(f"Failed to load model state: {e}")
                model = None
        else:
            logging.warning("Model file missing. Using fallback mode.")
            model = None

@app.route("/", methods=["GET"])
def index():
    initialize_app()
    titles = data["all_movies"]["title"].dropna().tolist() if not data["all_movies"].empty else []
    return render_template("index.html", titles=titles)

@app.route("/recommend", methods=["POST"])
def recommend():
    initialize_app()
    username = request.form.get("username")
    selected_titles = request.form.getlist("favorites")
    logging.info(f"User input: {username}, Favorites: {selected_titles}")

    if not username or not selected_titles:
        return redirect(url_for("index"))

    predictions = []

    if model:
        user_embedding = torch.nn.Parameter(torch.randn(100), requires_grad=True)
        optimizer = torch.optim.Adam([user_embedding], lr=0.01)

        for title in selected_titles:
            movie_row = data["all_movies"][data["all_movies"]["title"] == title]
            if not movie_row.empty:
                movie_id = movie_row.iloc[0]["movieId"]
                movie_idx = data["movie_to_index"].get(movie_id)
                if movie_idx is not None:
                    movie_tensor = torch.tensor([movie_idx], dtype=torch.long)
                    target_rating = torch.tensor([5.0])
                    for _ in range(30):
                        model.eval()
                        movie_emb = model.movie_embedding(movie_tensor).squeeze()
                        input_vec = torch.cat([user_embedding, movie_emb], dim=0)
                        pred = model.fc(input_vec.unsqueeze(0)).squeeze() + model.global_bias
                        loss = nn.functional.mse_loss(pred, target_rating)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

        for _, row in data["all_movies"].iterrows():
            mid = data["movie_to_index"].get(row["movieId"])
            if mid is not None and row["title"] not in selected_titles:
                try:
                    movie_emb = model.movie_embedding(torch.tensor([mid], dtype=torch.long))
                    sim = torch.nn.functional.cosine_similarity(user_embedding.unsqueeze(0), movie_emb).item()
                    predictions.append((row["title"], sim))
                except Exception as e:
                    logging.warning(f"Prediction failed for '{row['title']}': {e}")

        top_recs = sorted(predictions, key=lambda x: x[1], reverse=True)[:20]
        if not top_recs:
            top_recs = [(f"Dummy Movie {i+1}", round(1 - i * 0.05, 2)) for i in range(20)]
    else:
        top_recs = [(f"Dummy Movie {i+1}", round(1 - i * 0.05, 2)) for i in range(20)]

    return render_template("results.html", username=username, recommendations=top_recs, favorites=selected_titles)

@app.route("/feedback", methods=["POST"])
def feedback():
    username = request.form.get("username")
    selected_movies = request.form.get("favorites")
    user_feedback = request.form.get("feedback")

    logging.info(f"Feedback from {username}: {user_feedback}")

    try:
        with open("data/user_feedback.csv", "a", encoding="utf-8") as f:
            f.write(f"{username},{selected_movies},{user_feedback}\n")
    except Exception as e:
        logging.warning(f"Feedback saving failed: {e}")

    return render_template("thankyou.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Starting app on port {port}")
    app.run(host="0.0.0.0", port=port)
