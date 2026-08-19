"""
Chroma-Cinema
-------------
Pulls movie posters from TMDB, extracts each poster's dominant color palette
using OpenCV + K-Means, then aggregates palettes by genre to see if genres
have a characteristic "color signature" (e.g. horror = dark/red, comedy =
bright/saturated).

SETUP
-----
1. Get a free TMDB API key: https://www.themoviedb.org/settings/api
2. pip install requests opencv-python scikit-learn matplotlib numpy
3. Set your API key (choose one):
   - Windows PowerShell:  $env:TMDB_API_KEY="your_key_here"
   - Or just paste it directly into API_KEY_FALLBACK below (line ~30)
4. Run: python chroma_cinema.py

OUTPUT
------
- posters/           downloaded poster images (cached, won't re-download)
- palettes.csv        per-movie dominant colors + genre
- genre_palette.png    bar chart comparing average palette per genre
"""

import os
import csv
import time
import sys
import requests
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────
API_KEY_FALLBACK = ""   # paste your TMDB key here if you don't want to use env vars
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", API_KEY_FALLBACK)

BASE_URL = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p/w342"
POSTER_DIR = "posters"
N_COLORS = 5            # dominant colors extracted per poster
MOVIES_PER_GENRE = 20   # keep small for a fast first run; raise later for more signal

GENRES = {
    28: "Action",
    35: "Comedy",
    27: "Horror",
    18: "Drama",
    10749: "Romance",
    878: "Sci-Fi",
    16: "Animation",
    53: "Thriller",
}
# ─────────────────────────────────────────────────────────────────────────


def fetch_movies_for_genre(genre_id, n=MOVIES_PER_GENRE):
    """Grab top-rated/popular movies for a genre via TMDB discover endpoint."""
    movies = []
    page = 1
    while len(movies) < n:
        try:
            r = requests.get(
                f"{BASE_URL}/discover/movie",
                params={
                    "api_key": TMDB_API_KEY,
                    "with_genres": genre_id,
                    "sort_by": "popularity.desc",
                    "page": page,
                },
                timeout=10,
            )
        except requests.exceptions.RequestException as e:
            print(f"  ! network error on page {page}: {e}")
            break

        if r.status_code == 401:
            raise SystemExit(
                "TMDB rejected your API key (401 Unauthorized). "
                "Double-check TMDB_API_KEY is set correctly."
            )
        r.raise_for_status()

        results = r.json().get("results", [])
        if not results:
            break
        movies.extend(results)
        page += 1

    return movies[:n]


def download_poster(movie):
    """Download a poster if not already cached locally. Returns local path or None."""
    poster_path = movie.get("poster_path")
    if not poster_path:
        return None

    os.makedirs(POSTER_DIR, exist_ok=True)
    fname = os.path.join(POSTER_DIR, f"{movie['id']}.jpg")

    if os.path.exists(fname):
        return fname

    try:
        r = requests.get(IMG_BASE + poster_path, timeout=10)
        if r.status_code == 200:
            with open(fname, "wb") as f:
                f.write(r.content)
            return fname
    except requests.exceptions.RequestException:
        pass

    return None


def dominant_colors(image_path, k=N_COLORS):
    """Return k dominant RGB colors (0-255 ints), most dominant first."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 150))  # downsize for speed
    pixels = img.reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=k, n_init=4, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)

    counts = np.bincount(kmeans.labels_)
    order = np.argsort(-counts)
    return colors[order]


def main():
    if not TMDB_API_KEY:
        raise SystemExit(
            "No TMDB API key found.\n"
            "Either set an environment variable TMDB_API_KEY, "
            "or paste your key into API_KEY_FALLBACK at the top of this script."
        )

    rows = []
    genre_color_bins = defaultdict(list)

    for genre_id, genre_name in GENRES.items():
        print(f"[{genre_name}] fetching movie list...")
        movies = fetch_movies_for_genre(genre_id)
        print(f"[{genre_name}] got {len(movies)} movies, processing posters...")

        processed = 0
        for movie in movies:
            fname = download_poster(movie)
            if not fname:
                continue

            colors = dominant_colors(fname)
            if colors is None:
                continue

            top_color = colors[0]
            genre_color_bins[genre_name].append(top_color)

            rows.append({
                "title": movie.get("title", ""),
                "genre": genre_name,
                "dominant_rgb": tuple(top_color),
            })
            processed += 1
            time.sleep(0.05)  # be polite to the API

        print(f"[{genre_name}] processed {processed} posters\n")

    if not rows:
        raise SystemExit("No posters were processed — check your API key and network connection.")

    # ── write CSV ──
    with open("palettes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "genre", "dominant_r", "dominant_g", "dominant_b"])
        for row in rows:
            r, g, b = row["dominant_rgb"]
            writer.writerow([row["title"], row["genre"], r, g, b])
    print(f"Wrote palettes.csv ({len(rows)} movies total)")

    # ── aggregate + plot average dominant color per genre ──
    genre_names = list(genre_color_bins.keys())
    avg_colors = [np.mean(genre_color_bins[g], axis=0) / 255 for g in genre_names]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(genre_names, [1] * len(genre_names), color=avg_colors, width=0.6)
    ax.set_title("Average Dominant Poster Color by Genre — Chroma-Cinema")
    ax.set_yticks([])
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("genre_palette.png", dpi=150)
    print("Wrote genre_palette.png")
    print("\nDone. Open genre_palette.png and palettes.csv to see results.")


if __name__ == "__main__":
    main()
