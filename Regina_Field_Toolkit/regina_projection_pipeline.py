import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import umap
except ImportError:
    umap = None
    print("⚠️ UMAP module not found. Skipping UMAP projection.")

# === Configuration ===
DEFAULT_INPUT_CSV = "data/init/Calibration_Dataset.csv"
PLOT = True  # Set to False to disable plot generation

# === Step 1: Parse input arguments ===
input_csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_dir = "output/projection"
os.makedirs(output_dir, exist_ok=True)

output_csv = os.path.join(output_dir, f"regina_field_projection_output_{timestamp}.csv")
pca_plot_file = os.path.join(output_dir, f"pca_projection_plot_{timestamp}.png")
umap_plot_file = os.path.join(output_dir, f"umap_projection_plot_{timestamp}.png")
pca_score_plot = os.path.join(output_dir, f"pca_score_heatmap_{timestamp}.png")
umap_score_plot = os.path.join(output_dir, f"umap_score_heatmap_{timestamp}.png")
metadata_path = os.path.join(output_dir, f"metadata_{timestamp}.json")

# === Step 2: Load data ===
df = pd.read_csv(input_csv)

# === Step 3: Select features ===
feature_cols = [
    "MotifSum",
    "Entropy",
    "HilbertMag",
    "CompositeScore",
    "EnhancedCompositeScore",
    "RhythmicCompositeScore",
    "BoundaryTransitionIndex"
]

X = df[feature_cols].dropna()

# === Step 4: Standardize ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Step 5: PCA ===
if "PCA_X" not in df.columns or "PCA_Y" not in df.columns:
    pca = PCA(n_components=2, random_state=42)
    pca_proj = pca.fit_transform(X_scaled)
    df["PCA_X"] = np.nan
    df["PCA_Y"] = np.nan
    df.loc[X.index, "PCA_X"] = pca_proj[:, 0]
    df.loc[X.index, "PCA_Y"] = pca_proj[:, 1]
else:
    print("ℹ️ PCA projection already present — skipping PCA calculation.")

# === Step 6: UMAP ===
if umap is not None:
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
    umap_proj = reducer.fit_transform(X_scaled)
    df["UMAP_X"] = np.nan
    df["UMAP_Y"] = np.nan
    df.loc[X.index, "UMAP_X"] = umap_proj[:, 0]
    df.loc[X.index, "UMAP_Y"] = umap_proj[:, 1]
else:
    df["UMAP_X"] = np.nan
    df["UMAP_Y"] = np.nan

# === Step 7: Save output CSV ===
df.to_csv(output_csv, index=False)
print(f"✅ Projections saved: {output_csv}")

# === Step 8: Plotting ===
def plot_projection(x, y, title, out_file, color_col="PrimeStatus", cmap=None):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=color_col, palette=cmap, alpha=0.7, s=20)
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"🖼️ Plot saved: {out_file}")

def plot_heatmap(x, y, color_metric, title, out_file):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df[x], df[y], c=df[color_metric], cmap="viridis", alpha=0.8, s=20)
    plt.title(title)
    plt.colorbar(scatter, label=color_metric)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"🌡️ Heatmap saved: {out_file}")

if PLOT:
    plot_projection("PCA_X", "PCA_Y", "PCA Projection of Regina Field", pca_plot_file)
    if umap is not None:
        plot_projection("UMAP_X", "UMAP_Y", "UMAP Projection of Regina Field", umap_plot_file)
    if "Score" in df.columns:
        plot_heatmap("PCA_X", "PCA_Y", "Score", "PCA: Structural Score Heatmap", pca_score_plot)
        if umap is not None:
            plot_heatmap("UMAP_X", "UMAP_Y", "Score", "UMAP: Structural Score Heatmap", umap_score_plot)

# === Step 9: Save metadata JSON ===
metadata = {
    "timestamp": timestamp,
    "input_csv": input_csv,
    "output_csv": output_csv,
    "pca_plot_file": pca_plot_file if PLOT else None,
    "umap_plot_file": umap_plot_file if PLOT and umap is not None else None,
    "pca_score_heatmap": pca_score_plot if PLOT else None,
    "umap_score_heatmap": umap_score_plot if PLOT and umap is not None else None,
    "features_used": feature_cols,
    "rows_input": len(df),
    "rows_projected": len(X),
    "projection_settings": {
        "pca_components": 2,
        "umap_enabled": umap is not None,
        "umap_n_neighbors": 15,
        "umap_min_dist": 0.1,
        "umap_metric": "euclidean"
    }
}

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"📝 Metadata log saved: {metadata_path}")

# === Volatility Overlay ===
try:
    volatility_log = pd.read_csv("output/refinement/score_tracking_log.csv")
    volatility_map = (
        volatility_log.groupby("Candidate")["Score_curr"].std().reset_index().rename(columns={"Score_curr": "Volatility"})
    )
    df = df.merge(volatility_map, on="Candidate", how="left")

    def plot_volatility_overlay(x, y, score_col, size_col, title, out_file):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df, x=x, y=y,
            hue=score_col,
            size=size_col,
            sizes=(10, 200),
            palette="viridis",
            alpha=0.7,
            legend="brief"
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()
        print(f"🌀 Volatility overlay saved: {out_file}")

    if PLOT and "Volatility" in df.columns:
        pca_vol_file = os.path.join(output_dir, f"pca_volatility_overlay_{timestamp}.png")
        umap_vol_file = os.path.join(output_dir, f"umap_volatility_overlay_{timestamp}.png")
        plot_volatility_overlay("PCA_X", "PCA_Y", "Score", "Volatility", "PCA Score & Volatility", pca_vol_file)
        if umap is not None:
            plot_volatility_overlay("UMAP_X", "UMAP_Y", "Score", "Volatility", "UMAP Score & Volatility", umap_vol_file)
except Exception as e:
    print(f"⚠️ Volatility overlay failed: {e}")