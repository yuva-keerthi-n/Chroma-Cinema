# Chroma-Cinema

Do movie genres have a color signature? This project analyzes 200 real
movie posters (25 each across 8 genres) using OpenCV + K-Means to extract
dominant colors, then compares the results by genre.

**This version is fully offline — no API key, no internet needed.**
All 200 posters are already included in `posters/`.

## Setup

```bash
pip install opencv-python scikit-learn matplotlib numpy pandas
python chroma_cinema_local.py
```

That's it — no signup, no env vars, no network calls.

## What's included

- `posters/` — 200 real movie poster images (25 per genre)
- `selected.csv` — maps each poster to its genre
- `chroma_cinema_local.py` — the analysis script
- `palettes.csv` — pre-computed results (dominant RGB per poster)
- `genre_palette.png` — pre-computed chart of average color per genre

You can just look at `palettes.csv` and `genre_palette.png` right now
without running anything — or rerun the script yourself to regenerate
them (useful once you start editing the code).

## Result

Sorted brightest → darkest, average dominant poster color per genre:

| Genre     | R     | G     | B     |
|-----------|-------|-------|-------|
| Comedy    | 175.4 | 168.9 | 165.4 |
| Drama     | 145.0 | 145.4 | 143.7 |
| Romance   | 117.1 | 99.0  | 81.1  |
| Animation | 92.5  | 83.4  | 90.8  |
| Action    | 70.4  | 62.6  | 49.8  |
| Sci-Fi    | 65.2  | 58.5  | 60.0  |
| Horror    | 50.1  | 45.6  | 45.8  |
| Thriller  | 44.0  | 36.5  | 39.1  |

Comedy and Drama posters run noticeably brighter and more desaturated;
Horror and Thriller sit at the dark end, as you'd expect from genre
convention. Action and Sci-Fi cluster together in the middle-dark range.

## How it works

1. **Source data** — posters + genre labels from a public IMDB poster
   dataset (25 movies sampled per genre, favoring single/few-genre movies
   for a cleaner signal).
2. **Extract** — each poster is resized and clustered with K-Means (k=5)
   in RGB space; the largest cluster is taken as the "dominant color."
3. **Aggregate** — dominant colors are averaged within each genre and
   plotted as a bar chart.

## Ideas to extend

- Swap RGB clustering for HSV to separate hue from brightness/saturation
  (right now "brightness" and "color" are tangled together).
- Add more posters per genre (`N_PER_GENRE` in the original selection
  script) for a stronger statistical signal.
- Correlate palette "darkness" with genre and see if it predicts rating.
- Build a tiny classifier: given just a poster's palette, predict genre.
- Turn `genre_palette.png` into an interactive Plotly chart for your
  GitHub README.
- If you want to pull *live* new posters instead of the bundled set,
  see `chroma_cinema_live.py` (needs a free TMDB API key + working
  internet access — was blocked for us by local firewall/antivirus,
  so this offline version is the reliable path for now).
