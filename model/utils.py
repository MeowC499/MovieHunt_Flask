import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

def load_data():
    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")

    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    ratings["user_idx"] = user_encoder.fit_transform(ratings["userId"])
    ratings["movie_idx"] = movie_encoder.fit_transform(ratings["movieId"])

    user_to_index = dict(zip(ratings["userId"], ratings["user_idx"]))
    movie_to_index = dict(zip(ratings["movieId"], ratings["movie_idx"]))
    index_to_movie = {v: k for k, v in movie_to_index.items()}

    merged = pd.merge(movies, ratings, on="movieId")

    return {
        "ratings": ratings,
        "all_movies": movies,
        "user_to_index": user_to_index,
        "movie_to_index": movie_to_index,
        "index_to_movie": index_to_movie
    }

def get_recommendations(selected_titles, model, data, top_k=20):
    all_movies = data["all_movies"]
    movie_to_index = data["movie_to_index"]
    index_to_movie = data["index_to_movie"]
    device = torch.device("cpu")

    # Create a fake user embedding
    user_embedding = torch.nn.Parameter(torch.randn(100), requires_grad=True)
    optimizer = torch.optim.Adam([user_embedding], lr=0.01)

    # Train this pseudo-user on selected favorites
    for title in selected_titles:
        movie_row = all_movies[all_movies["title"] == title]
        if movie_row.empty:
            continue
        movie_id = movie_row["movieId"].values[0]
        movie_idx = movie_to_index.get(movie_id)
        if movie_idx is None:
            continue

        movie_tensor = torch.tensor([movie_idx], dtype=torch.long)
        target_rating = torch.tensor(5.0)

        for _ in range(30):
            movie_emb = model.movie_embedding(movie_tensor).squeeze()
            input_vec = torch.cat([user_embedding, movie_emb], dim=0)
            pred = model.fc(input_vec.unsqueeze(0)).squeeze() + model.global_bias
            loss = torch.nn.functional.mse_loss(pred, target_rating)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Score all movies
    recs = []
    for _, row in all_movies.iterrows():
        movie_id = row["movieId"]
        movie_idx = movie_to_index.get(movie_id)
        if movie_idx is None:
            continue
        movie_emb = model.movie_embedding(torch.tensor([movie_idx], dtype=torch.long))
        score = torch.nn.functional.cosine_similarity(user_embedding.unsqueeze(0), movie_emb).item()
        recs.append((row["title"], score))

    top_recs = sorted(recs, key=lambda x: x[1], reverse=True)[:top_k]
    return [{"title": title, "score": f"{score:.2f}"} for title, score in top_recs]
