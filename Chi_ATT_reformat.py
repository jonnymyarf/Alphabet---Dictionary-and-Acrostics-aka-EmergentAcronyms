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

# --- preprocessing constants ---
DEFAULT_TRUNCATE_TO_SHORTEST = False
DEFAULT_N_ = 2
DEFAULT_SIMPLE_ACRONYM_LENGTH = 2 + abs(DEFAULT_N_)
DEFAULT_INCLUDE_WORDS = set(w.upper() for w in nltk_words.words())
DEFAULT_EXCLUDE_WORDS = []
DEFAULT_N_CLUSTERS = 2
DEFAULT_RUNS = 300
DEFAULT_OPTIMIZE_FOR = "chi" #"cross_score"
DEFAULT_SEED = 0
DEFAULT_LINKAGE= "ward"#"single"#"average"#"complete"#"ward"
DEFAULT_REFORMAT = True


# --- dominance criteria defaults ---
DEFAULT_FILTER_ACRO_MAPS = False#True
DEFAULT_TOP_K_ACROS = None
DEFAULT_DOMINANCE = 1.0#5.0
DEFAULT_MIN_PERCENT = 0.4
DEFAULT_MORE_OR_LESS = "more"
DEFAULT_MORE_OR_LESS_COUNT = "more"
DEFAULT_EXCLUDE_ACRO = []
DEFAULT_THAN_ = 0

# --- print preferences ---
DEFAULT_PRINT_FOR = "O" #"cross_score"

def clean_text(text, include_words=DEFAULT_INCLUDE_WORDS, exclude_words=DEFAULT_EXCLUDE_WORDS):
    tokens = re.findall(r"[A-Za-z]+'?[A-Za-z]+|[A-Za-z]+", text)
    tokens = [t.replace("'", "").upper() for t in tokens]
    #tokens = [t for t in tokens if t in include_words]    
    tokens = [t for t in tokens if t not in exclude_words]    
    
    return tokens


def acro_map_simple(words, n=DEFAULT_SIMPLE_ACRONYM_LENGTH):
    acro_map = defaultdict(list)

    for i in range(len(words) - n + 1):
        # concatenate first letters
        acronym = "".join(words[i + j][0] for j in range(n))
        # concatenate full words
        acrostic = " ".join(words[i + j] for j in range(n))
        acro_map[acronym].append(acrostic)

    print("same speed")
    return acro_map


def load_text(file):
    try:
        if file.endswith(".json"):
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
                return data.get("text", "")
        else:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        print("Error loading:", file, e)
        return ""

def load_files(
        files,
        truncate_to_shortest=DEFAULT_TRUNCATE_TO_SHORTEST,
        simple_acronym_length=DEFAULT_SIMPLE_ACRONYM_LENGTH,
        include_words=DEFAULT_INCLUDE_WORDS,
        exclude_words=DEFAULT_EXCLUDE_WORDS,
    ):
    file_maps = {}
    min_len = float("inf")

    for file in files:
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            text = load_text(file)
            print("LEN:", len(text), "| FILE:", file[:60])

        words = clean_text(text, include_words, exclude_words)

        if len(words) < 500:   # ← you can tune this (100–300 range)
            print("SKIPPED (too short):", file, "| LEN:", len(words))
            continue
        print("WORDS LEN:", len(words))
        min_len = min(min_len, len(words))
        file_maps[file] = words

    # truncate to shortest file
    if truncate_to_shortest:
        for file in file_maps:
            file_maps[file] = file_maps[file][:min_len]

    # build acro maps
    acro_maps = {
        file: acro_map_simple(words, simple_acronym_length)
        for file, words in file_maps.items()
    }

    return acro_maps
    
def extract_acro_map_filtered(
    acro_map,
    acro_map2,
    dominance=DEFAULT_DOMINANCE,
    min_percent=DEFAULT_MIN_PERCENT,
    more_or_less=DEFAULT_MORE_OR_LESS,
    more_or_less_count=DEFAULT_MORE_OR_LESS_COUNT,
    exclude_acro=DEFAULT_EXCLUDE_ACRO,
    than_=DEFAULT_THAN_):
    if acro_map:
        acrostic_totals = {a: len(v) for a, v in acro_map.items()}
        max_freq = max(acrostic_totals.values())
    else:
        max_freq = 0

    acro_map_filtered = {}

    exclude_acro = [e.upper() for e in exclude_acro]
    for acronym, acrostics in acro_map.items():
        if any(letter.upper() in acronym for letter in exclude_acro):
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
                or max(v, o) / min(v, o) >= dominance
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
    
def cross_file_filter_acro_maps(
    acro_maps_by_file,
    dominance = DEFAULT_DOMINANCE,
    min_percent = DEFAULT_MIN_PERCENT,
    more_or_less = DEFAULT_MORE_OR_LESS,
    more_or_less_count = DEFAULT_MORE_OR_LESS_COUNT,
    exclude_acro = DEFAULT_EXCLUDE_ACRO,
    than_=DEFAULT_THAN_
    ):
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
            acro_map2,
            dominance=dominance,
            min_percent=min_percent,
            more_or_less=more_or_less,
            more_or_less_count=more_or_less_count,
            exclude_acro=exclude_acro,
            than_=than_
        )

        filtered_by_file[fid] = filtered
        filtered_count += 1
        print(f"ed {filtered_count}")

    return filtered_by_file

def reformat_file_acro_maps(acro_maps_by_file, n):
    reformatted = {}

    # --- safe sample ---
    try:
        first_file = next(iter(acro_maps_by_file.values()))
        first_acro = next(iter(first_file.keys()))
    except Exception:
        raise ValueError("Empty acro_maps_by_file")

    # --- validation ---
    import inspect
    if n == 0:
        raise ValueError(f"Line {inspect.currentframe().f_lineno}: n must not be 0")

    if n >= len(first_acro):
        raise ValueError(
            f"Line {inspect.currentframe().f_lineno}: n={n} must be < acronym length={len(first_acro)}"
        )

    # --- main transformation ---
    if n > 0:
        for file_map in acro_maps_by_file.values():
            for acro, acrostics in file_map.items():

                outer = acro[:n]        # e.g. "AA"
                inner = acro[n:]         # next letter, e.g. "A"

                if outer not in reformatted:
                    reformatted[outer] = {}

                if inner not in reformatted[outer]:
                    reformatted[outer][inner] = []

                reformatted[outer][inner].extend(acrostics)
    elif n < 0:
        for file_map in acro_maps_by_file.values():
            for acro, acrostics in file_map.items():

                outer = acro[n:]        # e.g. "AA"
                inner = acro[:n]         # next letter, e.g. "A"

                if outer not in reformatted:
                    reformatted[outer] = {}

                if inner not in reformatted[outer]:
                    reformatted[outer][inner] = []

                reformatted[outer][inner].extend(acrostics)

    return reformatted

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
    for f in files[:5]:
        print(f, len(acro_maps[f]))
        print("========================")
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

def score_kmeans_runs(
    counts,
    n_clusters=DEFAULT_N_CLUSTERS,
    runs=DEFAULT_RUNS,
    scale=True,
    optimize_for=DEFAULT_OPTIMIZE_FOR,
    linkage=DEFAULT_LINKAGE
):
    """
    Deterministic version:
    - no randomness
    - no multiple runs
    - same input → same output
    """

    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # --- scale (deterministic) ---
    if scale:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(counts.astype(float))
    else:
        X = counts.astype(float)

    # --- deterministic clustering ---
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )
    labels = model.fit_predict(X)

    # --- compute chi ---
    avg_yield, per_cluster_avg, cell_yields, O, E, R, C, T = \
        compute_chi_yield_for_partition(labels, counts)

    # --- deterministic cross score ---
    cross_scores = np.zeros_like(O, dtype=float)

    global_O = O.sum(axis=0)
    global_E = E.sum(axis=0)

    global_contrib = np.zeros_like(global_O)
    valid_global = global_E > 0
    global_contrib[valid_global] = (
        (global_O[valid_global] - global_E[valid_global]) /
        np.sqrt(global_E[valid_global])
    )

    v_global = global_O / (T + 1e-8)

    k = O.shape[0]

    for c in range(k):
        col_E = E[c]
        valid = col_E > 0

        contrib = np.zeros_like(global_O)
        contrib[valid] = (
            (O[c, valid] - col_E[valid]) /
            np.sqrt(col_E[valid])
        )

        v_cluster = O[c] / (T + 1e-8)

        for a in range(len(global_O)):
            if v_cluster[a] > 0 and v_global[a] > 0:
                c1 = contrib[a]
                c2 = global_contrib[a]

                denom = c1 + c2 if (c1 + c2) != 0 else 1e-8

                cross_scores[c, a] = (
                    min(v_cluster[a], v_global[a]) *
                    abs(c1 - c2) / denom
                )

    cross_score = cross_scores.mean()

    # --- identical output structure ---
    best_labels = labels.copy()
    best_info = {
        "avg_yield": avg_yield,
        "cross_score": cross_score,
        "cross_scores": cross_scores,
        "per_cluster_avg": per_cluster_avg,
        "cell_yields": cell_yields,
        "O": O,
        "E": E,
        "R": R,
        "C": C,
        "T": T,
        "seed": None,
        "linkage": linkage
    }

    return best_labels, best_info

def summarize_clusters(files, acros, counts, labels, best_info,
    top_k_acros=DEFAULT_TOP_K_ACROS, print_for=DEFAULT_PRINT_FOR,
    seed=DEFAULT_SEED):
    k = labels.max() + 1
    O = best_info["O"]
    E = best_info["E"]
    T = best_info["T"]
    cross_scores = best_info.get("cross_scores", None)
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
        if(print_for == "chi"):   
            top_idx = np.argsort(-contrib)[:top_k_acros]
        elif(print_for == "cross_score"):
            top_idx = np.argsort(-cross_scores[c])[:top_k_acros]
        elif(print_for == "O"):
            top_idx = np.argsort(-O[c])[:top_k_acros]
        else:
            print("invalid print_for")
            exit()
        top_list = []
        for idx in top_idx:
            if contrib[idx] <= 0:
                continue
            top_list.append({
                "acronym": acros[idx],
                "O": int(O[c, idx]),
                "E": float(col_E[idx]),
                "contrib": float(contrib[idx]),
                "cross_scores": float(cross_scores[c, idx])
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
    
def main_from_files(
    files,
    n_clusters=DEFAULT_N_CLUSTERS,
    runs=DEFAULT_RUNS,
    top_k_acros=DEFAULT_TOP_K_ACROS,
    truncate_to_shortest=DEFAULT_TRUNCATE_TO_SHORTEST,
    simple_acronym_length=DEFAULT_SIMPLE_ACRONYM_LENGTH,
    include_words=DEFAULT_INCLUDE_WORDS,
    exclude_words=DEFAULT_EXCLUDE_WORDS,
    optimize_for=DEFAULT_OPTIMIZE_FOR,
    seed=DEFAULT_SEED,
    filter_acro_maps=DEFAULT_FILTER_ACRO_MAPS,
    dominance=DEFAULT_DOMINANCE,
    min_percent=DEFAULT_MIN_PERCENT,
    more_or_less=DEFAULT_MORE_OR_LESS,
    more_or_less_count=DEFAULT_MORE_OR_LESS_COUNT,
    exclude_acro=DEFAULT_EXCLUDE_ACRO,
    than_=DEFAULT_THAN_,
    print_for=DEFAULT_PRINT_FOR,
    linkage=DEFAULT_LINKAGE,
    reformat=DEFAULT_REFORMAT,
    n_=DEFAULT_N_
):
    acro_maps = load_files(
        files,
        truncate_to_shortest=truncate_to_shortest,
        simple_acronym_length=simple_acronym_length,
        include_words=include_words,
        exclude_words=exclude_words,
    )
    
    if filter_acro_maps:
        acro_maps = cross_file_filter_acro_maps(
            acro_maps,
            dominance = dominance,
            min_percent = min_percent,
            more_or_less = more_or_less,
            more_or_less_count = more_or_less_count,
            exclude_acro = exclude_acro,
            than_ = than_
        )
    
    if reformat:
        acro_maps = reformat_file_acro_maps(acro_maps, n_)

    files_list, acros, counts = build_acro_count_matrix(acro_maps)

    labels, best_info = score_kmeans_runs(
        counts,
        n_clusters=n_clusters,
        runs=runs,
        optimize_for=optimize_for,
        linkage=linkage
    )

    summaries, files_by_cluster = summarize_clusters(
        files_list,
        acros,
        counts,
        labels,
        best_info,
        top_k_acros=top_k_acros,
        print_for=print_for,
        seed=seed
    )

    return {
        "avg_chi_yield": float(best_info["avg_yield"]),
        "per_cluster_avg": [float(x) for x in best_info["per_cluster_avg"].tolist()],
        "total_acrostics": int(best_info["T"]),
        "cluster_summaries": summaries,
        "files_by_cluster": files_by_cluster,
        "seed_used": best_info["seed"]#int(best_info["seed"]),
    }

def print_and_confirm_results(out):
    print("\n===== GLOBAL RESULT =====")
    print(f"Avg chi-yield: {out['avg_chi_yield']:.6f}")
    print(f"Total acrostics: {out['total_acrostics']}")
    print(f"Seed used: {out['seed_used']}")

    for c, files in out["files_by_cluster"].items():
        print(f"\n===== CLUSTER {c} =====")
        print(f"Files in cluster: {len(files)}")
        print(f"Cluster avg chi-yield: {out['per_cluster_avg'][c]:.6f}")
        for f in files:
            print(" ", f)
    '''
    print("\nType exactly the three lowercase letters yes, in that form and press enter to continue")
    if input("> ").strip().lower() != "yes":
        print("Aborted.")
        raise SystemExit
    
    print_cluster_acronym_stats(out)'''
        
def print_cluster_acronym_stats(out, print_for=DEFAULT_PRINT_FOR):
    d = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10, "K": 11, "L": 12, "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20, "U": 21, "V": 22, "W": 23, "X": 24, "Y": 25, "Z": 26}
    longest = 0
    att_combined = 0
    att_chi_combined = 0
    att_count_combined = 0
    for summary in out["cluster_summaries"]:
        item = summary['top_acronyms'][0]
        if print_for == "chi":
            longest_tmp = len(f"{  item['acronym']:>4} | chi≈{item['contrib']:.4f}")
        elif print_for == "cross_score":
            longest_tmp = len(f"{  item['acronym']:>4} | score≈{item['cross_scores']:.4f}")
        elif print_for == "O":
            longest_tmp = len(f"{  item['acronym']:>4} | score≈{item['O']:.4f}")
        if longest < longest_tmp:
            longest = longest_tmp
    for summary in out["cluster_summaries"]:
        att = 0
        att_chi = 0
        att_count = 0
        c = summary["cluster"]
        print(f"\n===== CLUSTER {c} ACRONYMS =====")
        
        for item in summary["top_acronyms"]:
            if print_for == "chi":
                printing = f"{  item['acronym']:>4} | chi≈{item['contrib']:.4f}"
            elif print_for == "cross_score":
                printing = f"{  item['acronym']:>4} | score≈{item['cross_scores']:.4f}"
            elif print_for == "O":
                printing = f"{  item['acronym']:>4} | score≈{item['O']:.4f}"
            printing += " " * (longest - len(printing))
            print(printing, end='')
            
            if (d[item['acronym'][0]] - d[item['acronym'][-1]]) < 0:
                # A->Z
                att += 1
                att_chi += item['O']
                att_count += 1
            elif (d[item['acronym'][0]] - d[item['acronym'][-1]]) > 0:
                # Z->A
                att -= 1
                att_chi -= item['O']
                att_count += 1
        att_combined += att
        att_chi_combined += att_chi
        att_count_combined += att_count
        print()
        print("Let A->Z be positive, and Z->A be negative")
        if att_count > 0:
            print(f"att = {att / att_count}")
            print(f"att_chi = {att_chi / att_count}")
        else:
            print("att = N/A")
            print("att_chi = N/A")
        print()
    if att_count > 0:
        print(f"att_combined = {att_combined / att_count_combined}")
        print(f"att_chi_combined = {att_chi_combined / att_count_combined}")
    else:
        print("att_combined = N/A")
        print("att_chi_combined = N/A")
    print()
    print

import os

def get_scrapy_file_paths(base_folder="Scrapy"):
    paths = []

    for source in os.listdir(base_folder):
        source_path = os.path.join(base_folder, source)

        if not os.path.isdir(source_path):
            continue

        for file in os.listdir(source_path):
            if file.endswith(".json"):
                paths.append(os.path.join(source_path, file))

    return paths
    
def build_relative_freq_dict(out):
    cats = {}

    for s in out["cluster_summaries"]:
        cluster_id = s["cluster"]

        counts = {}
        total = 0

        for item in s["top_acronyms"]:
            counts[item["acronym"]] = item["O"]
            total += item["O"]

        if total > 0:
            cats[cluster_id] = {
                a: c / total for a, c in counts.items()
            }
        else:
            cats[cluster_id] = {}

    return cats

def build_chi_dict(out):
    cats = {}
    for s in out["cluster_summaries"]:
        cluster_id = s["cluster"]
        d = {}
        for item in s["top_acronyms"]:
            d[item["acronym"]] = item["contrib"]
        cats[cluster_id] = d
    return cats

def categorical_full_permutation_similarity_freq_cosine_similarity(out1, out2):
    import itertools
    import numpy as np
    
    chi1 = build_chi_dict(out1)
    chi2 = build_chi_dict(out2)

    cats1 = build_relative_freq_dict(out1)
    cats2 = build_relative_freq_dict(out2)

    categories1 = list(cats1.keys())
    categories2 = list(cats2.keys())

    # ----- Determine full dimension K -----
    all_acros = set()
    for c in categories1:
        all_acros.update(cats1[c].keys())
    for c in categories2:
        all_acros.update(cats2[c].keys())

    K = len(all_acros)

    # ----- Estimate alpha from cats1 -----
    # Use average squared norm heuristic
    norms = []
    for c in categories1:
        v = np.array(list(cats1[c].values()))
        norms.append(np.sum(v ** 2))

    mean_norm = np.mean(norms)

    # Invert approximate Dirichlet identity:
    # E[||p||^2] ≈ (alpha + 1) / (K * alpha + 1)
    if K * mean_norm - 1 != 0:
        alpha = max((1 - mean_norm) / (K * mean_norm - 1), 1e-6)
    else:
        alpha = 0.01  # fallback

    best_avg = -float("inf")
    best_pairing = None
    
    vecs1 = {c: dict_to_vec(cats1[c], all_acros) for c in categories1}
    vecs2 = {c: dict_to_vec(cats2[c], all_acros) for c in categories2}

    for perm in itertools.permutations(categories1):

        remaining2 = categories2.copy()
        pairing = {}
        total = 0.0
        total_baseline = 0.0
        count = 0

        i = 0
                
        while i < len(perm) and remaining2:

            c1 = perm[i]

            best_val = -float("inf")
            best_match = None

            j = 0
            while j < len(remaining2):
                c2 = remaining2[j]
                # val = matrix_cosine_similarity(cats1[c1], cats2[c2])
                
                d1 = cats1[c1]
                d2 = cats2[c2]

                local_acros = set(d1.keys()) | set(d2.keys())

                val = 0.0
                for a in local_acros:
                    v1 = d1.get(a, 0.0)
                    v2 = d2.get(a, 0.0)
                    c1_val = chi1[c1].get(a, 0.0)
                    c2_val = chi2[c2].get(a, 0.0)
                    # simplest true overlap:
                    val += min(v1, v2)
                    # val += min(v1, v2) * abs(c1_val - c2_val) / (c1_val + c2_val)
                
                if val > best_val:
                    best_val = val
                    best_match = c2
                j += 1


            pairing[c1] = best_match
            total += best_val
            count += 1

            remaining2.remove(best_match)
            i += 1

        avg = total / count if count > 0 else 0.0

        if avg > best_avg:
            best_avg = avg
            best_pairing = pairing
    
    overlap_results = {}
    for c1, c2 in best_pairing.items():
        d1 = cats1[c1]
        d2 = cats2[c2]

        local_acros = set(d1.keys()) | set(d2.keys())

        scores = []
        for a in local_acros:
            v1 = d1.get(a, 0.0)
            v2 = d2.get(a, 0.0)
            
            c1_val = chi1[c1].get(a, 0.0)
            c2_val = chi2[c2].get(a, 0.0)

            # score = min(v1, v2)   # or your preferred formula
            score = min(v1, v2)# * abs(c1_val - c2_val) / (c1_val + c2_val)
            scores.append((a, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        overlap_results[(c1, c2)] = scores
    average_baseline = 0
    return best_avg, average_baseline, best_pairing, overlap_results

def dict_to_vec(d, all_acros):
    import numpy as np
    return np.array([d.get(a, 0.0) for a in all_acros], dtype=float)

def random_baseline(K, alpha, trials=500):
    import numpy as np

    vals = []
    for _ in range(trials):
        rng = np.random.RandomState(seed)
        r1 = rng.random.dirichlet(np.ones(K) * alpha)
        r2 = rng.random.dirichlet(np.ones(K) * alpha)

        n1 = np.linalg.norm(r1)
        n2 = np.linalg.norm(r2)

        if n1 == 0 or n2 == 0:
            vals.append(0.0)
        else:
            vals.append(float(np.dot(r1, r2) / (n1 * n2)))

    return float(np.mean(vals))
    
def print_overlap_results(overlap_results):
    d = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10, "K": 11, "L": 12, "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20, "U": 21, "V": 22, "W": 23, "X": 24, "Y": 25, "Z": 26}
    longest = 0
    longest_tmp = 0
    att_combined = 0
    att_score_combined = 0
    att_count_combined = 0
    # --- print per cluster pair ---
    for (c1, c2), scores in overlap_results.items():
        for a, score in scores[:33]:
            longest_tmp = len(f"{a:>4} | overlap≈{score:.4f}")
            if longest < longest_tmp:
                longest = longest_tmp
    
    for (c1, c2), scores in overlap_results.items():
        att = 0
        att_score = 0
        att_count = 0

        print(f"\n===== CLUSTER {c1} ↔ {c2} OVERLAP =====")

        for a, score in scores[:33]:
            printing = f"{a:>4} | overlap≈{score:.4f}"
            printing += " " * (longest - len(printing))
            print(printing, end='')

            if len(a) > 0:
                if (d[a[0]] - d[a[-1]]) < 0:
                    att += 1
                    att_score += score
                    att_count += 1
                elif (d[a[0]] - d[a[-1]]) > 0:
                    att -= 1
                    att_score -= score
                    att_count += 1

        att_combined += att
        att_score_combined += att_score
        att_count_combined += att_count

        print()
        print("Let A->Z be positive, and Z->A be negative")

        if att_count > 0:
            print(f"att = {att / att_count}")
            print(f"att_overlap = {att_score / att_count}")
        else:
            print("att = N/A")
            print("att_overlap = N/A")

        print()

    # --- combined stats ---
    if att_count_combined > 0:
        print(f"att_combined = {att_combined / att_count_combined}")
        print(f"att_overlap_combined = {att_score_combined / att_count_combined}")
    else:
        print("att_combined = N/A")
        print("att_overlap_combined = N/A")

    print()

def main():
    files = get_scrapy_file_paths()
    files_new = get_scrapy_file_paths(base_folder="Scrapy_new")
    
    '''[
        "Shakespere.txt", "Zizek_SOI.txt", "Hegel.txt",
        "ORWELL.txt", "HOBBES.txt", "PLATO.txt",
        "ARISTOTLE.txt", "Torah.txt", "Quran.txt",
        "Physicalists.txt", "Idealists.txt"
    ]#'''
    
    # --- preprocessing constants ---
    truncate_to_shortest = DEFAULT_TRUNCATE_TO_SHORTEST
    simple_acronym_length = DEFAULT_SIMPLE_ACRONYM_LENGTH
    include_words = DEFAULT_INCLUDE_WORDS
    exclude_words = DEFAULT_EXCLUDE_WORDS
    n_clusters = DEFAULT_N_CLUSTERS
    runs = DEFAULT_RUNS
    optimize_for = DEFAULT_OPTIMIZE_FOR
    seed = DEFAULT_SEED
    linkage=DEFAULT_LINKAGE
    reformat=DEFAULT_REFORMAT
    n_=DEFAULT_N_
    

    # --- dominance criteria defaults ---
    filter_acro_maps = DEFAULT_FILTER_ACRO_MAPS
    #filter_acro_maps = False # DEFAULT_FILTER_ACRO_MAPS,
    top_k_acros = DEFAULT_TOP_K_ACROS
    dominance=DEFAULT_DOMINANCE
    min_percent=DEFAULT_MIN_PERCENT
    more_or_less=DEFAULT_MORE_OR_LESS
    more_or_less_count=DEFAULT_MORE_OR_LESS_COUNT
    exclude_acro = DEFAULT_EXCLUDE_ACRO # ['T', 'MH']
    
    # --- print preferences ---
    print_for = DEFAULT_PRINT_FOR
    

    # exclude_words = ["a", "an", "the", "and", "but", "if", "nor", "or", "so", "yet", "as", "at", "by", "for", "in", "of", "off", "on", "per", "to", "up", "via"]

    out1 = main_from_files(
        files,
        n_clusters=n_clusters,
        runs=runs,
        top_k_acros=top_k_acros,
        truncate_to_shortest=truncate_to_shortest,
        simple_acronym_length=simple_acronym_length,
        include_words=include_words,
        exclude_words=exclude_words,
        optimize_for=optimize_for,
        seed=seed,
        filter_acro_maps=filter_acro_maps,
        dominance=dominance,
        min_percent=min_percent,
        more_or_less=more_or_less,
        more_or_less_count=more_or_less_count,
        exclude_acro = exclude_acro,
        print_for=print_for,
        linkage=linkage,
        reformat=reformat,
        n_ = n_
    )
    
    out2 = main_from_files(
        files_new,
        n_clusters=n_clusters,
        runs=runs,
        top_k_acros=top_k_acros,
        truncate_to_shortest=truncate_to_shortest,
        simple_acronym_length=simple_acronym_length,
        include_words=include_words,
        exclude_words=exclude_words,
        optimize_for=optimize_for,
        seed=seed,
        filter_acro_maps=filter_acro_maps,
        dominance=dominance,
        min_percent=min_percent,
        more_or_less=more_or_less,
        more_or_less_count=more_or_less_count,
        exclude_acro = exclude_acro,
        print_for=print_for,
        linkage=linkage,
        reformat=reformat,
        n_ = n_
    )
    
    print_and_confirm_results(out1)
    print_and_confirm_results(out2)
    
    print_cluster_acronym_stats(out1, print_for=print_for)
    print_cluster_acronym_stats(out2, print_for=print_for)
    
    a,b,c,d = categorical_full_permutation_similarity_freq_cosine_similarity(out1, out2)
    print_overlap_results(d)
    print(" "+str(a)+" "+str(b)+" "+str(c)+" ")
    print(f"Filtered: {filter_acro_maps}")

if __name__ == "__main__":
    main()
