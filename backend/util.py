# backend/utils.py

import random
import re
def load_movielens_items(file_path: str):
    """
    Load movies from the u.item file.
   """
    genre_labels = [
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    movies = []
    with open(file_path, "r", encoding="ISO-8859-1") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 24:
                continue #incase a row missing data 
            movie_id = parts[0]
            title = parts[1]
            release_date = parts[2]
            imdb_url = parts[4]

            genre_flags = parts[5:24]
            genres = [genre_labels[i] for i, flag in enumerate(genre_flags) if flag == "1"]

            movies.append({
                "movie_id": movie_id,
                "title": title,
                "release_date": release_date,
                "imdb_url": imdb_url,
                "genres": genres
            })
    return movies

def sample_movies(movies: list, sample_size: int = 5):
    """Randomly sample 'sample_size' movies from the list"""
    if len(movies) < sample_size:
        return movies
    return random.sample(movies, sample_size)
