"""
Chroma-Cinema — LOCAL VERSION (no internet/API needed)
-------------------------------------------------------
Analyzes movie posters already bundled in the posters/ folder,
extracts each poster's dominant colors via OpenCV + K-Means,
and aggregates them by genre to compare color signatures.

This version requires NO API key and NO network access —
everything runs on the 200 posters already included in posters/
(from a public IMDB poster dataset, 25 per genre across 8 genres).

SETUP
-----
pip install opencv-python scikit-learn matplotlib numpy pandas
python chroma_cinema_local.py

OUTPUT
------
- palettes.csv        per-movie dominant colors + genre
- genre_palette.png    bar chart of average palette per genre
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save the image
import matplotlib.pyplot as plt

SELECTED_CSV = "selected.csv"   # maps poster id -> genre -> file path
N_COLORS = 5


def dominant_colors(image_path, k=N_COLORS):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 150))
    pixels = img.reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=k, n_init=4, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    order = np.argsort(-counts)
    return colors[order]


def main():
    sel_df = pd.read_csv(SELECTED_CSV)
    rows = []

    print(f"Processing {len(sel_df)} posters...")
    for _, row in sel_df.iterrows():
        colors = dominant_colors(row["path"])
        if colors is None:
            continue
        top = colors[0]
        rows.append({
            "id": row["id"],
            "genre": row["genre"],
            "r": top[0], "g": top[1], "b": top[2],
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv("palettes.csv", index=False)
    print(f"Wrote palettes.csv ({len(result_df)} posters)")

    # aggregate + plot
    avg = result_df.groupby("genre")[["r", "g", "b"]].mean().reset_index()
    avg = avg.sort_values("r", ascending=False)
    colors = avg[["r", "g", "b"]].values / 255
    genres = avg["genre"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(genres, [1] * len(genres), color=colors, width=0.6,
           edgecolor="black", linewidth=0.5)
    ax.set_title("Average Dominant Poster Color by Genre — Chroma-Cinema")
    ax.set_yticks([])
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig("genre_palette.png", dpi=150)
    print("Wrote genre_palette.png")

    print("\nAverage RGB by genre (brightest to darkest):")
    print(avg.to_string(index=False))


if __name__ == "__main__":
    main()
