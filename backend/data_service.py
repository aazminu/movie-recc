# backend/data_service.py

import os
from backend.util import load_movielens_items, sample_movies # Utilitis to load movielens items and them pick random ones
from tmdb.tmdb_service import get_movie_details_by_title #TMDB API funcs
from config import TMDB_IMAGE_BASE_URL #URL for images

def get_enriched_movie_list(sample_size: int = 5, genre: str = "All"):
    """
   Return list of sampled and enriched movies, containging the title, genre, description and (maybe) a poster if possible
""" 
    ## Construct the path to the movielens file 
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "data", "ml-100k", "u.item")

    #Load movies from the file using itl func
    movies = load_movielens_items(file_path)

    # Filter the mvoies by genre
    if genre != "All":
        movies = [m for m in movies if genre in m.get("genres", [])]

    # Randomly sample a subset 
    sampled_movies = sample_movies(movies, sample_size)


    enriched_movies = []
    #'enrich' each movie with data from api 
    for movie in sampled_movies:
        title = movie.get("title")
        if not title:
            continue
            # Get the data from TMDB 
        tmdb_data = get_movie_details_by_title(title)
        if tmdb_data:
            movie["overview"] = tmdb_data.get("overview", "")
            poster_path = tmdb_data.get("poster_path")
            movie["poster_url"] = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
        else:
            movie["overview"] = ""
            movie["poster_url"] = None
        # add to list of movies 
        enriched_movies.append(movie)

    return enriched_movies
