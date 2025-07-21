import pandas as pd

# === Load Log ===
log_path = "output/refinement/score_tracking_log.csv"
df = pd.read_csv(log_path)

# === Compute Aggregates per Candidate ===
grouped = df.groupby("Candidate").agg({
    "Score_prev": "first",
    "Score_curr": ["last", "mean", "std", "min", "max"],
    "Delta_Score": ["sum", "mean", "max"],
    "BoundaryScore_prev": "first",
    "BoundaryScore_curr": ["last", "mean", "std", "min", "max"],
    "Delta_Boundary": ["sum", "mean", "max"],
    "Cycle": "count"
})

grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
grouped = grouped.reset_index()

# === Identify Volatile Candidates ===
volatile = grouped.sort_values(by="Score_curr_std", ascending=False).head(25)
volatile.to_csv("output/refinement/volatile_candidates.csv", index=False)

# === Candidates Near Threshold (hovering) ===
hovering = grouped[
    (grouped["Score_curr_last"] > 0.70) & (grouped["Score_curr_last"] < 0.80)
].sort_values(by="Score_curr_last", ascending=False)
hovering.to_csv("output/refinement/hovering_candidates.csv", index=False)

# === Candidates with Positive Trends ===
trending_up = grouped[grouped["Delta_Score_sum"] > 0].sort_values(by="Delta_Score_sum", ascending=False).head(25)
trending_up.to_csv("output/refinement/trending_up_candidates.csv", index=False)

print("✅ Score evolution analytics complete.")
print("- volatile_candidates.csv")
print("- hovering_candidates.csv")
print("- trending_up_candidates.csv")