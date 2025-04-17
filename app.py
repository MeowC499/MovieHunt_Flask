from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import torch
import torch.nn as nn
import os
import logging

from data_downloader import download_data
from model.ncf_model import NCF
from model.utils import load_data

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

data = None
model = None

def initialize_app():
    global data, model

    if data is None or model is None:
        logging.info("Initializing application: checking data and model...")

        download_data()  # Only downloads missing files, doesn’t duplicate
        logging.info("Loading data...")
        data = load_data()

        model_path = "data/ncf_model_state.pth"
        model = NCF(
            num_users=len(data['user_to_index']),
            num_movies=len(data['movie_to_index']),
            latent_dim=100,
            hidden_dims=[256, 128, 64],
            dropout_rate=0.3,
            use_batchnorm=True
        )

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            logging.info("Model loaded successfully.")
        else:
            logging.warning("Model not found. Running in dummy mode.")
            model = None


@app.route("/", methods=["GET"])
def index():
    initialize_app()
    return render_template("index.html", titles=data['all_movies']['title'].dropna().tolist())

@app.route("/recommend", methods=["POST"])
def recommend():
    initialize_app()
    username = request.form.get("username")
    selected_titles = request.form.getlist("favorites")
    logging.info(f"Received input - username: {username}, favorites: {selected_titles}")

    if not username or not selected_titles:
        return redirect(url_for('index'))

    predictions = []
    if model:
        user_embedding = torch.nn.Parameter(torch.randn(100), requires_grad=True)
        optimizer = torch.optim.Adam([user_embedding], lr=0.01)
        device = torch.device("cpu")

        for title in selected_titles:
            movie_row = data['all_movies'][data['all_movies']['title'] == title]
            if not movie_row.empty:
                movie_id = movie_row.iloc[0]['movieId']
                movie_idx = data['movie_to_index'].get(movie_id)
                logging.info(f"Processing movie '{title}' with ID {movie_id}, index: {movie_idx}")
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

        count_valid = 0
        for idx, row in data['all_movies'].iterrows():
            mid = data['movie_to_index'].get(row['movieId'])
            if mid is not None and row['title'] not in selected_titles:
                try:
                    movie_emb = model.movie_embedding(torch.tensor([mid], dtype=torch.long))
                    sim = torch.nn.functional.cosine_similarity(user_embedding.unsqueeze(0), movie_emb).item()
                    predictions.append((row['title'], sim))
                    count_valid += 1
                except Exception as e:
                    logging.warning(f"Error processing movie '{row['title']}': {e}")
        logging.info(f"Generated {count_valid} valid predictions.")

        top_recs = sorted(predictions, key=lambda x: x[1], reverse=True)[:20]
        if not top_recs:
            logging.warning("No recommendations found. Showing dummy fallback.")
            top_recs = [(f"Dummy Movie {i+1}", round(1 - i * 0.05, 2)) for i in range(20)]
    else:
        logging.info("Using dummy mode: generating placeholder recommendations")
        top_recs = [(f"Dummy Movie {i+1}", round(1 - i * 0.05, 2)) for i in range(20)]

    return render_template("results.html", username=username, recommendations=top_recs, favorites=selected_titles)

@app.route("/feedback", methods=["POST"])
def feedback():
    username = request.form.get("username")
    selected_movies = request.form.get("favorites")
    feedback = request.form.get("feedback")

    logging.info(f"Saving feedback for {username}: {feedback}")
    with open("data/user_feedback.csv", "a", encoding="utf-8") as f:
        f.write(f"{username},{selected_movies},{feedback}\n")

    return render_template("thankyou.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
