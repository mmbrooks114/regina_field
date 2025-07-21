import pandas as pd
from engine import io_utils, extrapolation

# Load scored + boundary-labeled primes
df = io_utils.load_csv("output/refinement/combined_with_boundary.csv")
known_primes = io_utils.load_csv("data/init/known_primes_up_to_10_million.csv")
false_elites = io_utils.load_csv("data/init/false_elites.csv")

# Select elite anchors
anchors = extrapolation.select_elite_anchors(df, top_n=100)

# Extrapolate new candidates
generated = extrapolation.extrapolate_candidates(anchors, n_samples=300)

# Score and filter them
scored = extrapolation.score_and_filter_candidates(generated)

# Validate
labeled = extrapolation.label_candidates(scored, known_primes, false_elites)

# Save output
io_utils.save_csv(labeled, "output/refinement/extrapolated_candidates.csv")
print("✅ Extrapolation cycle complete. Results saved.")