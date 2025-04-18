from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import torch
import torch.nn as nn
import os
import logging

from model.ncf_model import NCF

# Initialize Flask app
app = Flask(__name__)

# Setup Logging: because print() is for amateurs
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

data = None
model = None

def download_data():
    required_files = ["data/movies.csv", "data/ratings.csv", "data/user_feedback.csv"]
    for file in required_files:
        if not os.path.exists(file):
            logging.warning(f"Missing file: {file}. Upload required.")

def load_data():
    try:
        all_movies = pd.read_csv("data/movies.csv")
        if os.path.exists("data/user_feedback.csv"):
            user_feedback = pd.read_csv("data/user_feedback.csv")
        else:
            user_feedback = pd.DataFrame(columns=["username", "favorites", "feedback"])

        user_to_index = {user: idx for idx, user in enumerate(user_feedback["username"].unique())}
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
            "user_to_index": {},
            "movie_to_index": {}
        }

def initialize_app():
    global data, model

    if data is None or model is None:
        logging.info("Initializing app data and model...")
        download_data()
        data = load_data()

        model_path = "data/ncf_model_state.pth"
        model = NCF(
            num_users=len(data["user_to_index"]) or 1,
            num_movies=len(data["movie_to_index"]) or 1,
            latent_dim=100,
            hidden_dims=[256, 128, 64],
            dropout_rate=0.3,
            use_batchnorm=True
        )

        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
                model.eval()
                logging.info("Model loaded successfully from checkpoint.")
            except Exception as e:
                logging.warning(f"Failed to load model state: {e}")
                model = None
        else:
            logging.warning("Model file not found. Running in fallback mode.")
            model = None

@app.route("/", methods=["GET"])
def index():
    initialize_app()
    movie_titles = data["all_movies"]["title"].dropna().tolist() if not data["all_movies"].empty else []
    logging.info(f"Rendering index page with {len(movie_titles)} movie titles.")
    return render_template("index.html", titles=movie_titles)

@app.route("/recommend", methods=["POST"])
def recommend():
    initialize_app()
    username = request.form.get("username")
    selected_titles = request.form.getlist("favorites")
    logging.info(f"User input received - Username: {username}, Favorites: {selected_titles}")

    if not username or not selected_titles:
        logging.warning("Username or favorites missing. Redirecting to index.")
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
            movie_id = row["movieId"]
            movie_title = row["title"]
            if movie_title not in selected_titles:
                movie_idx = data["movie_to_index"].get(movie_id)
                if movie_idx is not None:
                    try:
                        movie_emb = model.movie_embedding(torch.tensor([movie_idx], dtype=torch.long))
                        sim = torch.nn.functional.cosine_similarity(user_embedding.unsqueeze(0), movie_emb).item()
                        predictions.append((movie_title, sim))
                    except Exception as e:
                        logging.warning(f"Error processing '{movie_title}': {e}")

        top_recs = sorted(predictions, key=lambda x: x[1], reverse=True)[:20]
        logging.info(f"Generated {len(top_recs)} recommendations for user {username}.")

        if not top_recs:
            logging.warning("No valid recommendations found. Returning dummy data.")
            top_recs = [(f"Dummy Movie {i+1}", round(1 - i * 0.05, 2)) for i in range(20)]
    else:
        logging.info("Model unavailable. Using dummy recommendations.")
        top_recs = [(f"Dummy Movie {i+1}", round(1 - i * 0.05, 2)) for i in range(20)]

    return render_template("results.html", username=username, recommendations=top_recs, favorites=selected_titles)

@app.route("/feedback", methods=["POST"])
def feedback():
    username = request.form.get("username")
    selected_movies = request.form.get("favorites")
    user_feedback = request.form.get("feedback")

    logging.info(f"Received feedback from {username}: {user_feedback}")

    try:
        with open("data/user_feedback.csv", "a", encoding="utf-8") as f:
            f.write(f"{username},{selected_movies},{user_feedback}\n")
        logging.info("Feedback saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save feedback: {e}")

    return render_template("thankyou.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port)
