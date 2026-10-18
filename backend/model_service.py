from collections import defaultdict
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

DEFAULT_GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def get_genre_cols(genre_path: str = "data/ml-100k/u.genre") -> List[str]:
    if pd is not None and os.path.exists(genre_path):
        genres_df = pd.read_csv(genre_path, sep="|", names=["genre", "genre_id"], header=None)
        return genres_df["genre"].dropna().tolist()
    if os.path.exists(genre_path):
        cols = []
        with open(genre_path, "r", encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("|")
                if parts and parts[0]:
                    cols.append(parts[0])
        if cols:
            return cols
    return list(DEFAULT_GENRES)


if nn is not None:
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
else:  # pragma: no cover
    MovieRecModel = None


_MODEL_CACHE: Optional[Tuple[Any, Dict[int, int], List[str]]] = None


def load_model_and_metadata(
    train_csv_path: str = "train.csv",
    model_path: str = "model.pth",
):
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if torch is None or pd is None or not (os.path.exists(train_csv_path) and os.path.exists(model_path)):
        return None, {}, get_genre_cols()

    train_df = pd.read_csv(train_csv_path)
    genre_cols_local = train_df.columns[3:].tolist()
    num_users = train_df["user_idx"].nunique()
    num_items = train_df["item_idx"].nunique()
    num_genres = len(genre_cols_local)

    model = MovieRecModel(num_users, num_items, num_genres, emb_dim=32)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    item_encoder = {int(item): idx for idx, item in enumerate(train_df["item_idx"].unique())}
    _MODEL_CACHE = (model, item_encoder, genre_cols_local)
    return _MODEL_CACHE


def _load_catalog_items(genre_cols_local: List[str], item_path: str = "data/ml-100k/u.item") -> List[Dict[str, Any]]:
    if not os.path.exists(item_path):
        return []
    items: List[Dict[str, Any]] = []
    with open(item_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 5:
                continue
            try:
                movie_id = int(parts[0])
            except ValueError:
                continue
            title = parts[1]
            raw_genres = parts[5 : 5 + len(genre_cols_local)]
            genre_vec = [float(x) if x.strip() in ("0", "1") else 0.0 for x in raw_genres]
            if len(genre_vec) < len(genre_cols_local):
                genre_vec.extend([0.0] * (len(genre_cols_local) - len(genre_vec)))
            items.append({"movie_id": movie_id, "title": title, "genres": np.asarray(genre_vec, dtype=np.float32)})
    return items


def recommend_for_group(
    group_ratings: List[Dict[str, Any]],
    candidate_movies: Optional[Iterable[Dict[str, Any]]] = None,
    predictor: Optional[Callable[[int, int, np.ndarray], float]] = None,
    top_k: int = 3,
    fairness_lambda: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Compute residual-bias-corrected group movie recommendations with optional
    Pareto-fair least-misery aggregation:
      score(i) = (1 - fairness_lambda) * mean_u(s(u, i)) + fairness_lambda * min_u(s(u, i))
    Deduplicates group members so users who submitted more ratings do not
    disproportionately weight the group utility.
    """
    if not group_ratings or top_k <= 0:
        return []
    lam = max(0.0, min(1.0, float(fairness_lambda)))

    model, item_encoder, genre_cols_local = load_model_and_metadata()
    catalog = list(candidate_movies) if candidate_movies is not None else _load_catalog_items(genre_cols_local)
    if not catalog:
        return []

    movie_rows: Dict[int, Dict[str, Any]] = {}
    for item in catalog:
        mid = int(item["movie_id"])
        genres = np.asarray(item.get("genres", np.zeros(len(genre_cols_local))), dtype=np.float32)
        movie_rows[mid] = {"movie_id": mid, "title": str(item["title"]), "genres": genres}

    # Estimate per-user genre affinity from observed group ratings when neural model is offline
    user_genre_num: Dict[str, np.ndarray] = {}
    user_genre_den: Dict[str, np.ndarray] = {}
    for r_entry in group_ratings:
        uid = str(r_entry.get("user_id"))
        rmid = int(r_entry.get("movie_id"))
        rval = float(r_entry.get("rating"))
        if rmid in movie_rows:
            gvec = movie_rows[rmid]["genres"]
            if uid not in user_genre_num:
                user_genre_num[uid] = np.zeros_like(gvec, dtype=np.float32)
                user_genre_den[uid] = np.zeros_like(gvec, dtype=np.float32)
            user_genre_num[uid] = user_genre_num[uid] + gvec * (rval - 3.0)
            user_genre_den[uid] = user_genre_den[uid] + np.abs(gvec)

    def _predict(train_user_idx: int, movie_id: int, genres: np.ndarray, user_id: Optional[str] = None) -> float:
        if predictor is not None:
            return float(predictor(train_user_idx, movie_id, genres))
        if model is not None and torch is not None and movie_id in item_encoder:
            item_idx = item_encoder[movie_id]
            with torch.no_grad():
                g_t = torch.tensor([genres.tolist()], dtype=torch.float32)
                u_t = torch.tensor([train_user_idx], dtype=torch.long)
                i_t = torch.tensor([item_idx], dtype=torch.long)
                return float(model(u_t, i_t, g_t).item())
        # Deterministic content-based baseline incorporating per-user genre affinity
        active = float(np.sum(genres))
        genre_bonus = float(np.dot(genres, np.linspace(0.1, 0.5, num=len(genres)))) / max(active, 1.0)
        affinity = 0.0
        if user_id and user_id in user_genre_num and active > 0:
            denom = np.where(user_genre_den[user_id] > 0, user_genre_den[user_id], 1.0)
            user_pref = user_genre_num[user_id] / denom
            affinity = float(np.dot(genres, user_pref)) / active
        return 3.0 + genre_bonus + 0.5 * affinity

    user_residuals: Dict[str, List[float]] = defaultdict(list)
    unique_members: Dict[str, int] = {}

    for rating_entry in group_ratings:
        user_id = str(rating_entry.get("user_id"))
        rated_movie_id = int(rating_entry.get("movie_id"))
        actual_rating = float(rating_entry.get("rating"))
        train_user_idx = int(rating_entry.get("train_user_idx", 0))
        unique_members[user_id] = train_user_idx

        if rated_movie_id in movie_rows:
            row = movie_rows[rated_movie_id]
            predicted_rating = _predict(train_user_idx, rated_movie_id, row["genres"], user_id=user_id)
            user_residuals[user_id].append(actual_rating - predicted_rating)

    user_bias: Dict[str, float] = {
        uid: (sum(res) / len(res) if res else 0.0)
        for uid, res in ((u, user_residuals.get(u, [])) for u in unique_members)
    }

    rated_movie_ids = {int(r["movie_id"]) for r in group_ratings}
    predictions: List[Tuple[str, float]] = []

    for mid, row in movie_rows.items():
        if mid in rated_movie_ids:
            continue
        candidate_scores = []
        for user_id, train_user_idx in unique_members.items():
            base_score = _predict(train_user_idx, mid, row["genres"], user_id=user_id)
            adjusted_score = base_score + user_bias.get(user_id, 0.0)
            candidate_scores.append(adjusted_score)

        if candidate_scores:
            mean_score = sum(candidate_scores) / len(candidate_scores)
            min_score = min(candidate_scores)
            blended_score = (1.0 - lam) * mean_score + lam * min_score
            predictions.append((row["title"], blended_score))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return [{"title": title, "score": round(score, 2)} for title, score in predictions[:top_k]]

def compute_group_consensus_metrics(member_scores: List[float]) -> Dict[str, float]:
    """Compute mean utility, egalitarian min utility, and disagreement variance for a group item."""
    if not member_scores:
        return {"mean_score": 0.0, "min_score": 0.0, "disagreement_std": 0.0, "consensus_index": 0.0}
    vals = [float(x) for x in member_scores]
    mean_v = sum(vals) / len(vals)
    min_v = min(vals)
    var_v = sum((x - mean_v) ** 2 for x in vals) / len(vals)
    std_v = var_v ** 0.5
    consensus = max(0.0, mean_v - 0.5 * std_v)
    return {
        "mean_score": round(mean_v, 6),
        "min_score": round(min_v, 6),
        "disagreement_std": round(std_v, 6),
        "consensus_index": round(consensus, 6),
    }
