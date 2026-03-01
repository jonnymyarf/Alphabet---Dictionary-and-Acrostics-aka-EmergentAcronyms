import argparse
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import math
import json
import os


import re
from collections import defaultdict
import nltk
from nltk.corpus import words as nltk_words
from collections import Counter

nltk.download("words")

ENGLISH_WORDS = set(w.upper() for w in nltk_words.words())

global exclude
global more_or_less
global than_
global DOMINANCE
global min_percent
global more_or_less_count

exclude = [] # exclude = ["T", "S"]
more_or_less = "more"
than_ = 0
DOMINANCE = 5 / 1
min_percent = 0.1
more_or_less_count = "more"

def clean_text(text):
    tokens = re.findall(r"[A-Za-z]+'?[A-Za-z]+|[A-Za-z]+", text)
    tokens = [t.replace("'", "").upper() for t in tokens]
    tokens = [t for t in tokens if t in ENGLISH_WORDS]
    return tokens


def acro_map_simple(words, n=5):
    acro_map = defaultdict(list)

    for i in range(len(words) - n + 1):
        # concatenate first letters
        acronym = "".join(words[i + j][0] for j in range(n))
        # concatenate full words
        acrostic = " ".join(words[i + j] for j in range(n))
        acro_map[acronym].append(acrostic)

    print("same speed")
    return acro_map


def load_files(files, n=5):
    file_maps = {}
    min_len = float("inf")

    for file in files:
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        words = clean_text(text)
        min_len = min(min_len, len(words))
        file_maps[file] = words

    # truncate to shortest file
    for file in file_maps:
        file_maps[file] = file_maps[file][:min_len]

    # build acro maps
    acro_maps = {
        file: acro_map_simple(words, n)
        for file, words in file_maps.items()
    }

    return acro_maps
    
def extract_acro_map_filtered(acro_map, acro_map2):
    if acro_map:
        acrostic_totals = {a: len(v) for a, v in acro_map.items()}
        max_freq = max(acrostic_totals.values())
    else:
        max_freq = 0

    acro_map_filtered = {}

    for acronym, acrostics in acro_map.items():
        if any(letter in acronym for letter in exclude):
            continue

        counts = Counter(acrostics)

        # Frequency filtering
        if more_or_less == "more":
            filtered = {k: v for k, v in counts.items() if v > than_}
        elif more_or_less == "less":
            filtered = {k: v for k, v in counts.items() if v < than_}
        else:
            filtered = counts.copy()

        total = sum(filtered.values())

        # DOMINANCE filter
        other_counts = Counter(acro_map2.get(acronym, []))
        filtered = {
            k: v
            for k, v in filtered.items()
            if (
                (o := other_counts.get(k, 0)) == 0
                or max(v, o) / min(v, o) >= DOMINANCE
            )
        }

        threshold = max_freq * min_percent

        if more_or_less_count == "more":
            if total <= threshold:
                continue
        elif more_or_less_count == "less":
            if total <= threshold:
                continue

        # Expand back into acro_map form
        acro_map_filtered[acronym] = [
            k for k, v in filtered.items() for _ in range(v)
        ]
    return acro_map_filtered
    
def cross_file_filter_acro_maps(acro_maps_by_file):
    """
    For each file:
    1. Multiply its acro_map by (N - 1)
    2. Combine all other files into acro_map2
    3. Run extract_acro_map_filtered
    """
    filtered_by_file = {}

    file_ids = list(acro_maps_by_file.keys())
    n_files = len(file_ids)
    
    filtered_count = 0
    for fid in file_ids:
        print("filter",end='',flush=True)
        acro_map = acro_maps_by_file[fid]

        # 1. Multiply current file by (N - 1)
        multiplier = n_files - 1
        acro_map_scaled = {
            acronym: acrostics * multiplier
            for acronym, acrostics in acro_map.items()
        }

        # 2. Build acro_map2 from all other files
        acro_map2 = {}
        for other_fid in file_ids:
            if other_fid == fid:
                continue

            for acronym, acrostics in acro_maps_by_file[other_fid].items():
                acro_map2.setdefault(acronym, []).extend(acrostics)

        # 3. Run filter
        filtered = extract_acro_map_filtered(
            acro_map_scaled,
            acro_map2
        )

        filtered_by_file[fid] = filtered
        filtered_count += 1
        print(f"ed {filtered_count}")

    return filtered_by_file

def build_acro_count_matrix(acro_maps):
    """
    acro_maps: dict filename -> acro_map (mapping acronym -> list(acrostics))
    returns:
      files: list of filenames
      acros: list of all acronyms (sorted)
      counts: numpy array shape (num_files, num_acros) of integer counts
    """
    files = sorted(acro_maps.keys())
    all_acros = set()
    for f in files:
        all_acros.update(acro_maps[f].keys())
    acros = sorted(all_acros)
    idx = {a:i for i,a in enumerate(acros)}
    counts = np.zeros((len(files), len(acros)), dtype=np.int64)
    for i,f in enumerate(files):
        amap = acro_maps[f]
        for a,candidates in amap.items():
            counts[i, idx[a]] = len(candidates)
    return files, acros, counts

def compute_chi_yield_for_partition(labels, counts):
    """
    labels: array-like of length num_files with values in 0..k-1
    counts: numpy array shape (num_files, num_acros)
    returns:
      avg_yield: float average over all valid cells of sqrt((O-E)^2 / E)
      per_cluster_avg: array length k, cluster-level averages
      cell_yields: array shape (k, m) with values or zeros where E==0
      O, E, R, C, T for inspection
    """
    labels = np.asarray(labels)
    num_files, m = counts.shape
    k = labels.max() + 1
    # observed counts per cluster x acronym
    O = np.zeros((k, m), dtype=np.float64)
    for c in range(k):
        rows = (labels == c)
        if rows.sum() > 0:
            O[c, :] = counts[rows].sum(axis=0)
    R = O.sum(axis=1)  # sum per cluster (row totals)
    C = O.sum(axis=0)  # sum per acronym (col totals)
    T = O.sum()
    if T == 0:
        return 0.0, np.zeros(k), np.zeros((k,m)), O, None, R, C, T
    # expected E = R[c] * C[a] / T
    E = np.outer(R, C) / float(T)
    mask = E > 0
    # chi-yield per cell
    cell_yields = np.zeros_like(O, dtype=float)

    valid = (E > 0) & mask
    cell_yields[valid] = np.sqrt((O[valid] - E[valid])**2 / E[valid])
    # average across all valid cells
    valid_count = mask.sum()
    avg_yield = float(cell_yields.sum() / valid_count) if valid_count > 0 else 0.0
    # cluster averages (ignore acronyms where E==0 for that row)
    per_cluster_avg = np.zeros(k)
    for c in range(k):
        valid = mask[c]
        if valid.sum() > 0:
            per_cluster_avg[c] = cell_yields[c, valid].mean()
        else:
            per_cluster_avg[c] = 0.0
    return avg_yield, per_cluster_avg, cell_yields, O, E, R, C, T

def score_kmeans_runs(counts, n_clusters, runs=20, scale=True, random_seed=0):
    """
    Run KMeans 'runs' times with different seeds and return the best clustering
    according to avg chi-yield.
    """
    best = None
    best_labels = None
    # scale features (columns) because acronyms vary wildly
    if scale:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(counts.astype(float))
    else:
        X = counts.astype(float)
    rng = np.random.RandomState(random_seed)
    for i in range(runs):
        seed = rng.randint(0, 2**31-1)
        km = KMeans(n_clusters=n_clusters, n_init=1, random_state=int(seed))
        labels = km.fit_predict(X)
        avg_yield, per_cluster_avg, cell_yields, O, E, R, C, T = compute_chi_yield_for_partition(labels, counts)
        if (best is None) or (avg_yield > best):
            best = avg_yield
            best_labels = labels.copy()
            best_info = {
                "avg_yield": avg_yield,
                "per_cluster_avg": per_cluster_avg,
                "cell_yields": cell_yields,
                "O": O,
                "E": E,
                "R": R,
                "C": C,
                "T": T,
                "seed": seed
            }
    return best_labels, best_info

def summarize_clusters(files, acros, counts, labels, best_info, top_k_acros=10):
    k = labels.max() + 1
    O = best_info["O"]
    E = best_info["E"]
    T = best_info["T"]
    summaries = []
    for c in range(k):
        # For cluster c, compute contribution score for each acronym:
        # (O - E) / sqrt(E) is directional; we can sort by absolute magnitude or positive only
        # We'll prefer acronyms with largest positive contribution: (O - E)/sqrt(E)
        col_E = E[c]
        valid = col_E > 0
        contrib = np.zeros(len(acros))
        contrib[valid] = (O[c, valid] - col_E[valid]) / np.sqrt(col_E[valid])
        # get top positive contributions
        top_idx = np.argsort(-contrib)[:top_k_acros]
        top_list = []
        for idx in top_idx:
            if contrib[idx] <= 0:
                continue
            top_list.append({
                "acronym": acros[idx],
                "O": int(O[c, idx]),
                "E": float(col_E[idx]),
                "contrib": float(contrib[idx])
            })
        summaries.append({
            "cluster": int(c),
            "num_files": int((labels==c).sum()),
            "cluster_avg_yield": float(best_info["per_cluster_avg"][c]),
            "top_acronyms": top_list
        })
    # Also provide file lists per cluster
    files_by_cluster = {int(c): [] for c in range(k)}
    for f, label in zip(files, labels):
        files_by_cluster[int(label)].append(f)
    return summaries, files_by_cluster

def main_from_files(files, n_clusters=5, runs=30, top_k_acros=all):
    # Use your load_files to get acro maps. Expect load_files(files, n=5) -> acro_maps
    acro_maps = load_files(files, n=5)  # uses your load_files; truncates to shortest file
    files_list, acros, counts = build_acro_count_matrix(acro_maps)
    labels, best_info = score_kmeans_runs(counts, n_clusters, runs=runs)
    summaries, files_by_cluster = summarize_clusters(files_list, acros, counts, labels, best_info, top_k_acros)
    out = {
        "avg_chi_yield": float(best_info["avg_yield"]),
        "per_cluster_avg": [float(x) for x in best_info["per_cluster_avg"].tolist()],
        "total_acrostics": int(best_info["T"]),
        "cluster_summaries": summaries,
        "files_by_cluster": files_by_cluster,
        "seed_used": int(best_info["seed"])
    }
    return out

def main():
    files = [
        "Shakespere.txt", "Zizek_SOI.txt", "Hegel.txt",
        "ORWELL.txt", "HOBBES.txt", "PLATO.txt",
        "ARISTOTLE.txt", "Torah.txt", "Quran.txt",
        "Physicalists.txt", "Idealists.txt"
    ]

    n_clusters = 3
    runs = 300

    acro_maps = load_files(files, n=5)
    acro_maps = cross_file_filter_acro_maps(acro_maps) # critical filter, to separate signal from noise
    files_list, acros, counts = build_acro_count_matrix(acro_maps)

    labels, best_info = score_kmeans_runs(
        counts,
        n_clusters=n_clusters,
        runs=runs
    )

    print("\n===== GLOBAL RESULT =====")
    print(f"Avg chi-yield: {best_info['avg_yield']:.6f}")
    print(f"Total acrostics: {int(best_info['T'])}")
    print(f"Seed used: {best_info['seed']}")

    k = labels.max() + 1
    for c in range(k):
        print(f"\n===== CLUSTER {c} =====")
        print(f"Files in cluster: {(labels == c).sum()}")
        print(f"Cluster avg chi-yield: {best_info['per_cluster_avg'][c]:.6f}")
        for f, lab in zip(files_list, labels):
            if lab == c:
                print(" ", f)

if __name__ == "__main__":
    main()