def validate_candidates(candidates, known_primes, false_elites):
    candidates["IsPrime"] = candidates["Candidate"].isin(known_primes)
    candidates["IsFalseElite"] = candidates["Candidate"].isin(false_elites)
    return candidates