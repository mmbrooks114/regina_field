def rank_candidates(df, score_col="Score"):
    return df.sort_values(by=score_col, ascending=False)