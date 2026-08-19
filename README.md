# Chroma-Cinema

Do movie genres have a color signature? This project pulls movie posters from
TMDB, extracts each poster's dominant colors with OpenCV + K-Means, and
aggregates the palettes by genre — e.g. does horror skew dark/red, does
comedy skew bright/saturated, does sci-fi skew blue?

## Setup

```bash
pip install requests opencv-python-headless scikit-learn matplotlib numpy --break-system-packages
export TMDB_API_KEY="your_key_here"   # free key: https://www.themoviedb.org/settings/api
python chroma_cinema.py
```

## Output

- `posters/` — downloaded poster images
- `palettes.csv` — per-movie dominant RGB + genre
- `genre_palette.png` — bar chart of the average dominant color per genre

## How it works

1. **Fetch** — TMDB `/discover/movie` pulls top-rated movies per genre (8 genres, 20 movies each by default — bump `MOVIES_PER_GENRE` for more signal).
2. **Extract** — each poster is resized and clustered with K-Means (k=5) in RGB space to find its dominant colors.
3. **Aggregate** — dominant colors are averaged within each genre and plotted.

## Ideas to extend

- Swap RGB clustering for HSV to separate hue from brightness/saturation.
- Correlate palette "darkness" with genre and see if it predicts rating or box office.
- Build a tiny classifier: given just a poster's palette, predict genre.
- Turn `genre_palette.png` into an interactive Plotly chart for the GitHub README.
