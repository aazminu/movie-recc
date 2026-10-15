import requests
import re
from config import TMDB_API_KEY, TMDB_IMAGE_BASE_URL


# In-memory cache to store API responses and avoid duplicate calls.
cache = {}

def extract_title_and_year(full_title: str):
    """
    Extract the base title and the year from the movie title.
    Example: "Usual Suspects, The (1995)" returns ("Usual Suspects, The", "1995").
    If no year is found, returns the title and None.
    """
    match = re.search(r"^(.*)\s+\((\d{4})\)$", full_title)
    if match:
        base_title = match.group(1).strip()
        year = match.group(2)
        return base_title, year
    return full_title, None

def get_movie_details_by_title(full_title: str):
    """
    Query TMDb for movie details using the title (with the year extracted from parentheses).
    Uses the /search/movie endpoint.
    """
    base_title, year = extract_title_and_year(full_title)
    # Create a unique cache key based on title and year.
    cache_key = f"{base_title}_{year}" if year else base_title
    if cache_key in cache:
        return cache[cache_key]
    
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": base_title
    }
    if year:
        params["year"] = year
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])
        if results:
            details = results[0]  # Take the first match
            cache[cache_key] = details
            return details
    return None

# if __name__ == "__main__":
#     title = "Usual Suspects, The (1995)"
#     details = get_movie_details_by_title(title)
#     if details:
#         print("Title:", details.get("title"))
#         print("Overview:", details.get("overview"))
#         poster_path = details.get("poster_path")
#         if poster_path:
#             print("Poster URL:", f"{TMDB_IMAGE_BASE_URL}{poster_path}")
#         else:
#             print("No poster available.")
#     else:
#         print("Movie not found.")