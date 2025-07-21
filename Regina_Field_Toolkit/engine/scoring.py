def score_candidate(row):
    """
    Compute a weighted score for a given candidate row based on structural features.
    Weights can be tuned based on feedback or training.

    Features used:
    - MotifSum: structural complexity
    - Entropy: local/global signal coherence
    - HilbertMag: harmonic signal strength
    - BoundaryTransitionIndex: field transition behavior
    """

    motif = row.get("MotifSum", 0) or 0
    entropy = row.get("Entropy", 0) or 0
    hilbert = row.get("HilbertMag", 0) or 0
    boundary = row.get("BoundaryTransitionIndex", 0) or 0

    # Normalized scoring weights (can be tuned)
    score = (
        0.4 * motif +
        0.25 * entropy +
        0.25 * hilbert +
        0.10 * (1.0 - abs(boundary))  # stability bonus
    )
    return score