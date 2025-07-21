import pandas as pd
import os
from scoring import score_candidate

# === Prime Type Definitions ===
prime_type_definitions = {
    "Mersenne": [2**p - 1 for p in [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127]],
    "Fermat": [2**(2**n) + 1 for n in range(5)],  # Known Fermat primes: n=0 to 4
    "SophieGermain": [p for p in range(2, 10000) if all(pd.Series([p, 2*p + 1]).apply(pd.api.types.is_integer))],
    "Twin": [],  # Populated below
    "Safe": [],  # Populated below
    "Super": [],  # Populated below
    "Palindromic": [],  # Populated below
    "Wilson": [5, 13, 563],
    "Wieferich": [1093, 3511],
    "Chen": [],  # Skip unless you want deeper logic
    "Cousin": [],  # Skip unless you want deeper logic
    "Emirp": [],  # Skip unless you want deeper logic
    "Gaussian": [],  # Optional: primes ≡ 1 (mod 4) + 2
    "Regular": [],  # Optional: defined differently in different contexts
    "Constellation": [],  # Typically requires group-based logic
}

# === Load Dataset ===
df = pd.read_csv("data/init/Calibration_Dataset.csv")
df.set_index("Number", inplace=True)
all_numbers = df.index.tolist()

# === Utilities ===
def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n**0.5)+1, 2):
        if n % i == 0: return False
    return True

def build_additional_prime_types():
    primes = [n for n in all_numbers if is_prime(n)]
    primes_set = set(primes)

    for p in primes:
        # Twin primes
        if (p + 2 in primes_set) or (p - 2 in primes_set):
            prime_type_definitions["Twin"].append(p)
        # Safe primes
        if (p - 1) // 2 in primes_set:
            prime_type_definitions["Safe"].append(p)
        # Super primes
        if primes.index(p) + 1 in primes_set:
            prime_type_definitions["Super"].append(p)
        # Palindromic primes
        if str(p) == str(p)[::-1]:
            prime_type_definitions["Palindromic"].append(p)

build_additional_prime_types()

# === Score and Output ===
os.makedirs("output/refinement/prime_type_signatures", exist_ok=True)

for ptype, plist in prime_type_definitions.items():
    scored = []
    for p in plist:
        if p in df.index:
            row = df.loc[p]
            score = score_candidate(row)
            record = row.to_dict()
            record.update({
                "Candidate": p,
                "Score": score,
                "IsPrime": 1,
                "PrimeType": ptype
            })
            scored.append(record)

    if scored:
        out_df = pd.DataFrame(scored)
        out_path = f"output/refinement/prime_type_signatures/scored_{ptype.lower()}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"✅ Saved {ptype} prime scores to: {out_path}")
    else:
        print(f"⚠️ No entries found for: {ptype}")
