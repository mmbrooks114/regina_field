from engine import io_utils, scoring, extrapolation, ranking, validation, model
import pandas as pd

# === Load Data ===
init_known = io_utils.load_csv("data/init/known_primes_up_to_10_million.csv")
init_false = io_utils.load_csv("data/init/false_elites.csv")
calibration = io_utils.load_csv("data/init/Calibration_Dataset.csv")

# === Score Known Primes ===
calibration_indexed = calibration.set_index("Number")
known_candidates = init_known["Candidate"]
scored_known = []

for n in known_candidates:
    if n in calibration_indexed.index:
        row = calibration_indexed.loc[n]
        score = scoring.score_candidate(row)
        entry = row.to_dict()
        entry.update({"Candidate": n, "Score": score, "IsPrime": 1})
        scored_known.append(entry)

scored_known_df = pd.DataFrame(scored_known)

if scored_known_df.empty:
    print("⚠️ Warning: No scored known primes found. Check dataset alignment.")
else:
    io_utils.save_csv(scored_known_df, "output/refinement/scored_primes.csv")

# === Score False Elites ===
false_candidates = init_false["Candidate"]
scored_false = []

for n in false_candidates:
    if n in calibration_indexed.index:
        row = calibration_indexed.loc[n]
        score = scoring.score_candidate(row)
        entry = row.to_dict()
        entry.update({"Candidate": n, "Score": score, "IsPrime": 0})
        scored_false.append(entry)

scored_false_df = pd.DataFrame(scored_false)

if scored_false_df.empty:
    print("⚠️ Warning: No scored false elites found. Check dataset alignment.")
else:
    io_utils.save_csv(scored_false_df, "output/refinement/scored_false_elites.csv")

# === Combine and Train Boundary Model ===
combined = pd.concat([scored_known_df, scored_false_df], ignore_index=True)

# ✅ Check class balance before training
label_counts = combined["IsPrime"].value_counts()
print("Training set class distribution:", label_counts.to_dict())

if len(label_counts) < 2:
    raise ValueError("❌ Boundary model training failed: dataset must contain both prime (1) and non-prime (0) classes.")

# === Train and Apply Boundary Model ===
boundary_model = model.train_boundary_model(combined)
combined_scored = model.apply_boundary_model(combined, boundary_model)
io_utils.save_csv(combined_scored, "output/refinement/combined_with_boundary.csv")

print("✅ Scoring and model training complete. Boundary scores saved.")
