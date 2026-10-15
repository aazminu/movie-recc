import torch
import torch.nn as nn
import pandas as pd
from collections import defaultdict

#Get genre columns from the file
def get_genre_cols():
    #Split on thje '|' 
    genres_df = pd.read_csv("data/ml-100k/u.genre", sep="|", names=["genre", "genre_id"], header=None)
    return genres_df["genre"].tolist()

genre_cols = get_genre_cols()

# Ensure same architecture is used as in training 
class MovieRecModel(nn.Module):
    def __init__(self, num_users, num_items, num_genres, emb_dim=32):
        super(MovieRecModel, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.genre_fc = nn.Linear(num_genres, emb_dim)
        self.fc1 = nn.Linear(emb_dim * 3, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, user, item, genre):
        user_vec = self.user_emb(user)
        item_vec = self.item_emb(item)
        genre_vec = self.genre_fc(genre)
        x = torch.cat([user_vec, item_vec, genre_vec], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x.squeeze()

# Function to load the trained model and metadata
def load_model_and_metadata():
    train_df = pd.read_csv("train.csv")
    genre_cols_local = train_df.columns[3:].tolist()
    num_users = train_df["user_idx"].nunique()
    num_items = train_df["item_idx"].nunique()
    num_genres = len(genre_cols_local)
    
    model = MovieRecModel(num_users, num_items, num_genres, emb_dim=32)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    return model, train_df, genre_cols_local

model, train_df, genre_cols_local = load_model_and_metadata() # load the model once so it can be used multiple times 

# This maps the movie_id  to a unique index used in the model
item_encoder = {int(item): idx for idx, item in enumerate(train_df["item_idx"].unique())}

def recommend_for_group(group_ratings):
    """
    group_ratings: a list of dicts, each with keys:
       - 'user_id'
       - 'movie_id'
       - 'rating'
       - 'train_user_idx'  (assigned when the user joined the room)
    Steps :
      1. Load candidate movies from the u.item file.
      2. Exclude movies already rated by the group.
      3. For each group member, compute a bias adjustment from their residuals.
         For each rated movie that exists in u.item, compute:
              residual = actual rating - predicted rating.
         Average these residuals to obtain a bias for that user.
      4. For each candidate movie, compute predictions for each group member,
         add the corresponding bias adjustment, and then average the adjusted scores.
      5. Return the top candidates (top 3) based on these adjusted average scores.
    """

    # Load candidate movies from u.item
    items_df = pd.read_csv("data/ml-100k/u.item", sep="|", header=None, encoding="latin-1")
    movie_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + genre_cols_local
    items_df = items_df.iloc[:, :len(movie_columns)]
    items_df.columns = movie_columns

    # Build a mapping from movie_id to its row for faster lookup
    movie_rows = {int(row['movie_id']): row for _, row in items_df.iterrows()}

    # Compute bias adjustments for each group member
    # For each rating they submitted, compute residual = actual rating - predicted rating
    user_residuals = defaultdict(list)
    print("\n--- Computing Residuals ---", flush=True)
    for rating_entry in group_ratings:
        user_id = rating_entry.get("user_id")
        rated_movie_id = rating_entry.get("movie_id")
        actual_rating = rating_entry.get("rating")
        train_user_idx = rating_entry.get("train_user_idx", 0)

        if rated_movie_id in movie_rows:
            row = movie_rows[rated_movie_id]
            try:
                item_id = int(row['movie_id'])
            except Exception:
                continue
            item_idx = item_encoder.get(item_id)
            if item_idx is None:
                continue

            # Build genre tensor for the rated movie- inputs 
            genre_values = row[genre_cols_local].values.astype(float)
            genre_tensor = torch.tensor([genre_values], dtype=torch.float32)
            user_tensor = torch.tensor([train_user_idx], dtype=torch.long)
            item_tensor = torch.tensor([item_idx], dtype=torch.long)

            # predict and compute the residuals 
            with torch.no_grad():
                predicted_rating = model(user_tensor, item_tensor, genre_tensor).item()
            residual = actual_rating - predicted_rating
            user_residuals[user_id].append(residual)
            print(f"User {user_id}, Rated Movie {rated_movie_id}: Actual {actual_rating}, Predicted {predicted_rating:.4f}, Residual {residual:.4f}", flush=True)

    # Store the residuals 
    print("\n--- User Residuals ---", flush=True)
    for user_id, residual_list in user_residuals.items():
        print(f"User {user_id} residuals: {residual_list}", flush=True)

    # Average residuals to get a bias for each user
    user_bias = {}
    for rating_entry in group_ratings:
        user_id = rating_entry.get("user_id")
        residuals = user_residuals.get(user_id, [])
        if residuals:
            user_bias[user_id] = sum(residuals) / len(residuals)
        else:
            user_bias[user_id] = 0.0

    print("\n--- User Biases ---", flush=True)
    print(user_bias, flush=True)

    # Exclude movies already rated by the group
    rated_movie_ids = set(r["movie_id"] for r in group_ratings)
    candidates = items_df[~items_df['movie_id'].isin(rated_movie_ids)].copy()

    predictions = []
    print("\n--- Computing Predictions for Candidates ---", flush=True)
    # For each candidate, compute an adjusted prediction for every group member and average
    for _, row in candidates.iterrows():
        try:
            item_id = int(row['movie_id'])
        except Exception:
            continue
        item_idx = item_encoder.get(item_id)
        if item_idx is None:
            continue

        # Build genre tensor for candidate movie.
        genre_values = row[genre_cols_local].values.astype(float)
        genre_tensor = torch.tensor([genre_values], dtype=torch.float32)

        candidate_scores = []
        for r in group_ratings:
            train_user_idx = r.get("train_user_idx", 0)
            user_id = r.get("user_id")
            user_tensor = torch.tensor([train_user_idx], dtype=torch.long)
            item_tensor = torch.tensor([item_idx], dtype=torch.long)
            with torch.no_grad():
                base_score = model(user_tensor, item_tensor, genre_tensor).item()
            # Add the user's bias adjustment.
            bias = user_bias.get(user_id, 0.0)
            adjusted_score = base_score + bias
            candidate_scores.append(adjusted_score)
        
        # Log candidate details and per-user scores
        print(f"Candidate '{row['title']}' (Movie ID {item_id}): Scores {candidate_scores}", flush=True)
        
        if candidate_scores:
            avg_score = sum(candidate_scores) / len(candidate_scores)
            predictions.append((row['title'], avg_score))

    # Sort candidates by average adjusted score and take the top 3
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_3 = predictions[:3]
    print("\nTop 3 Predicted Movies (Adjusted):", flush=True)
    for title, score in top_3:
        print(f"  {title}: {score:.4f}", flush=True)

    return [
        {"title": title, "score": round(score, 2)}
        for title, score in top_3
    ]