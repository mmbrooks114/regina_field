import pandas as pd
from pathlib import Path
from engine import io_utils, extrapolation, model, scoring

# === Load Data ===
combined_path = "output/refinement/combined_with_boundary.csv"
log_path = "output/refinement/score_tracking_log.csv"

combined = io_utils.load_csv(combined_path)
known_primes = io_utils.load_csv("data/init/known_primes_up_to_10_million.csv")
false_elites = io_utils.load_csv("data/init/false_elites.csv")

# === Step 1: Extrapolate ===
anchors = extrapolation.select_elite_anchors(combined, top_n=100)
generated = extrapolation.extrapolate_candidates(anchors, n_samples=300)
scored = extrapolation.score_and_filter_candidates(generated)
labeled = extrapolation.label_candidates(scored, known_primes, false_elites)

# === Step 2: Integrate New Candidates ===
existing_ids = set(combined["Candidate"])
new_valid = labeled[~labeled["Candidate"].isin(existing_ids)]

if not new_valid.empty:
    print(f"🧬 {len(new_valid)} new extrapolated candidates identified. Updating model...")
    combined_updated = pd.concat([combined, new_valid], ignore_index=True)

    # Retrain and apply model
    boundary_model = model.train_boundary_model(combined_updated)
    combined_updated = model.apply_boundary_model(combined_updated, boundary_model)

    # Save full updated dataset
    io_utils.save_csv(combined_updated, combined_path)
    io_utils.save_csv(new_valid, "output/refinement/newly_integrated_candidates.csv")

    # === Score Tracking ===
    try:
        previous_scores = combined[["Candidate", "Score", "BoundaryScore"]].copy()
        current_scores = combined_updated[["Candidate", "Score", "BoundaryScore"]].copy()

        merged = previous_scores.merge(current_scores, on="Candidate", suffixes=("_prev", "_curr"))
        merged["Delta_Score"] = merged["Score_curr"] - merged["Score_prev"]
        merged["Delta_Boundary"] = merged["BoundaryScore_curr"] - merged["BoundaryScore_prev"]
        merged["Cycle"] = pd.Timestamp.now().isoformat()

        if Path(log_path).exists():
            existing_log = pd.read_csv(log_path)
            merged = pd.concat([existing_log, merged], ignore_index=True)

        io_utils.save_csv(merged, log_path)
        print("📊 Score evolution logged.")
    except Exception as e:
        print(f"⚠️ Score tracking failed: {e}")

    print("✅ Enrichment cycle complete. Boundary model updated.")
else:
    print("🔁 No new candidates passed filtering. No retraining necessary.")