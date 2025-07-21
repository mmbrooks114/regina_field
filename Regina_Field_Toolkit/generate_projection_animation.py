import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import os

# === Load and Prepare Data ===
log_path = "output/refinement/score_tracking_log.csv"
projection_path = "output/refinement/combined_with_projections.csv"
output_dir = "output/projection"
os.makedirs(output_dir, exist_ok=True)

try:
    df_proj = pd.read_csv(projection_path)
    df_log = pd.read_csv(log_path)
except Exception as e:
    raise RuntimeError(f"Error loading input files: {e}")

# Validate necessary columns
required_cols = ["Candidate", "Cycle", "PCA_X", "PCA_Y", "UMAP_X", "UMAP_Y", "Score"]
for col in required_cols:
    if col not in df_proj.columns and col not in df_log.columns:
        raise ValueError(f"Missing required column: {col}")

# Merge log and projections
df = df_proj.merge(df_log, on="Candidate", how="inner")
df = df[required_cols].dropna()

# Save history
history_path = os.path.join(output_dir, "projection_history.csv")
df.to_csv(history_path, index=False)

# === Animated PCA & UMAP Scatter Plots ===
def animate_projection(df, x_col, y_col, label, out_gif):
    fig, ax = plt.subplots(figsize=(8, 6))
    cycles = sorted(df["Cycle"].unique())

    def update(frame):
        ax.clear()
        cycle_df = df[df["Cycle"] == frame]
        sns.scatterplot(data=cycle_df, x=x_col, y=y_col, hue="Score", palette="viridis", s=30, ax=ax, legend=False)
        ax.set_title(f"{label} - Cycle {frame}")
        ax.set_xlim(df[x_col].min() - 1, df[x_col].max() + 1)
        ax.set_ylim(df[y_col].min() - 1, df[y_col].max() + 1)

    ani = animation.FuncAnimation(fig, update, frames=cycles, repeat=False)
    ani.save(out_gif, writer="pillow", fps=1)
    print(f"🎥 Animation saved: {out_gif}")

# === Progression Trails ===
def plot_trails(df, x_col, y_col, label, out_file):
    plt.figure(figsize=(8, 6))
    for cand in df["Candidate"].unique():
        sub = df[df["Candidate"] == cand]
        plt.plot(sub[x_col], sub[y_col], alpha=0.5)
        plt.scatter(sub[x_col].iloc[-1], sub[y_col].iloc[-1], c="red", s=10)
    plt.title(f"{label} (Progression Trails)")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"🛤️ Trail plot saved: {out_file}")

# === Run ===
animate_projection(df, "PCA_X", "PCA_Y", "PCA Projection", os.path.join(output_dir, "pca_animation.gif"))
animate_projection(df, "UMAP_X", "UMAP_Y", "UMAP Projection", os.path.join(output_dir, "umap_animation.gif"))
plot_trails(df, "PCA_X", "PCA_Y", "PCA Projection", os.path.join(output_dir, "pca_trails.png"))
plot_trails(df, "UMAP_X", "UMAP_Y", "UMAP Projection", os.path.join(output_dir, "umap_trails.png"))