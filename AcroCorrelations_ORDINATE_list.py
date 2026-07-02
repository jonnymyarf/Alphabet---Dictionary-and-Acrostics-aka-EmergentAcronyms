# =========================
# Acrostic Average-Case Method v0.2
# =========================

from datasets import load_dataset
from collections import Counter, defaultdict
import random
import statistics
import re


ordinate_list = ["philosophy", "revolutionary", "happiness", "good"]
ORDINATE = "philosophy"
ACRO = "/\\"
RUN_MANY_NULLS = False

# -------------------------
# STEP 1: data
# -------------------------

#"""
#EMERGENCY BACKUP:
typical_sentences = [
    "The black stone apple fell.",
    "Before sunrise, birds sang across the quiet field.",
    "Simple tests help reveal hidden structure in language.",
    "The project compares ordinary English with frequency matched random English.",
    "Better source data will make the acrostic statistics more meaningful."
]

print("Loaded emergency fallback sentences:", len(typical_sentences))


import re
from pathlib import Path


def extract_sentences_from_text(text):
    """
    #Extract sentences from a block of text.

    #Splits on sentence-ending punctuation: . ! ?
    #Keeps the punctuation attached to the sentence.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Match sentence-like chunks ending in ., !, or ?
    sentences = re.findall(r"[^.!?]+[.!?]", text)

    # Clean leading/trailing spaces
    return [s.strip() for s in sentences]


def extract_sentences_from_file(file_path):
    """
    #Read a text file and return a list of sentences.
    """
    text = Path(file_path).read_text(encoding="utf-8")
    return extract_sentences_from_text(text)


# TEXT DOCUMENT OF DATA:::::
# TEXT DOCUMENT OF DATA:::::
# TEXT DOCUMENT OF DATA:::::
# TEXT DOCUMENT OF DATA:::::
# TEXT DOCUMENT OF DATA:::::
# TEXT DOCUMENT OF DATA:::::
# TEXT DOCUMENT OF DATA:::::
# TEXT DOCUMENT OF DATA:::::
# TEXT DOCUMENT OF DATA:::::
# TEXT DOCUMENT OF DATA:::::
typical_sentences = extract_sentences_from_file("GEORG WILHELM FRIEDRICH HEGEL.txt")
#typical_sentences = extract_sentences_from_file("TheLongestTextEver.txt")

# i, sentence in enumerate(sentences):
#    print(i, sentence)
#"""

# -------------------------
# Tokenization
# -------------------------

WORD_RE = re.compile(r"[A-Za-z]+")

def tokenize(sentence):
    return WORD_RE.findall(sentence.lower())

def first_initial(word):
    return word[0].upper()


# -------------------------
# STEP 1.5: Build null dataset
# -------------------------

def build_frequency_table(sentences):
    words = []

    for sentence in sentences:
        words.extend(tokenize(sentence))

    counts = Counter(words)

    vocab = list(counts.keys())
    weights = list(counts.values())

    return vocab, weights


def generate_null_sentences(real_sentences, vocab, weights):
    """
    Preserves sentence length, but replaces each word independently
    accRg to empirical unigram frequencies.
    """

    null_sentences = []

    for sentence in real_sentences:
        real_words = tokenize(sentence)
        k = len(real_words)

        sampled_words = random.choices(vocab, weights=weights, k=k)
        null_sentences.append(" ".join(sampled_words))

    return null_sentences


vocab, weights = build_frequency_table(typical_sentences)

null_sentences = generate_null_sentences(
    real_sentences=typical_sentences,
    vocab=vocab,
    weights=weights
)




# -------------------------
# STEP 2: Extract acro events
# -------------------------

def extract_acro_events(
    sentences,
    dataset_kind,
    source_type,
    n,
    null_run_id=None
):
    rows = []

    for sentence_id, sentence in enumerate(sentences):
        words = tokenize(sentence)

        # Need n acro words plus one next word.
        for i in range(0, len(words) - n):
            acro_words = words[i:i+n]
            next_word = words[i+n]

            acro = "".join(first_initial(w) for w in acro_words)
            next_initial = first_initial(next_word)

            rows.append({
                "dataset_kind": dataset_kind,      # typical or null
                "source_type": source_type,        # leipzig, gutenberg, wikipedia, etc.
                "null_run_id": null_run_id,        # None for typical
                "sentence_id": sentence_id,
                "word_position": i,
                "n": n,
                "acro": acro,
                "next_word": next_word,
                "next_initial": next_initial,
            })

    return rows


typical_events = extract_acro_events(
    sentences=typical_sentences,
    dataset_kind="typical",
    source_type="leipzig",
    n=2
)

null_events = extract_acro_events(
    sentences=null_sentences,
    dataset_kind="null",
    source_type="leipzig",
    n=2,
    null_run_id=1
)




# -------------------------
# STEP 3: Score acro → next initial
# -------------------------

def score_acro_next_initial(rows, acro, next_initial):
    relevant = [r for r in rows if r["acro"] == acro]

    if len(relevant) == 0:
        return None

    hits = [r for r in relevant if r["next_initial"] == next_initial]

    return {
        "acro": acro,
        "next_initial": next_initial,
        "count_acro": len(relevant),
        "count_hits": len(hits),
        "rate": len(hits) / len(relevant),
    }


real_bs_a = score_acro_next_initial(
    typical_events,
    acro="BS",
    next_initial="A"
)

null_bs_a = score_acro_next_initial(
    null_events,
    acro="BS",
    next_initial="A"
)

print("Real BS→A:", real_bs_a)
print("Null BS→A:", null_bs_a)





# -------------------------
# STEP 4: Many null worlds
# -------------------------

def run_many_nulls(real_sentences, vocab, weights, n, acro, next_initial, runs=100):
    null_scores = []

    for run_id in range(runs):
        null_sentences = generate_null_sentences(real_sentences, vocab, weights)

        null_events = extract_acro_events(
            sentences=null_sentences,
            dataset_kind="null",
            source_type="leipzig",
            n=n,
            null_run_id=run_id
        )

        score = score_acro_next_initial(null_events, acro, next_initial)

        if score is not None:
            null_scores.append(score["rate"])

    return null_scores


def z_score(real_value, null_values):
    mu = statistics.mean(null_values)
    sd = statistics.stdev(null_values)

    return {
        "real": real_value,
        "null_mean": mu,
        "null_sd": sd,
        "z": (real_value - mu) / sd if sd != 0 else None
    }

if(RUN_MANY_NULLS):
    null_bs_a_rates = run_many_nulls(
        real_sentences=typical_sentences,
        vocab=vocab,
        weights=weights,
        n=2,
        acro="BS",
        next_initial="A",
        runs=100
    )

    real_bs_a_rate = real_bs_a["rate"]

    result = z_score(real_bs_a_rate, null_bs_a_rates)

    print(result)




# -------------------------
# STEP B1: SBERT semantic ordinate setup
# -------------------------

from sentence_transformers import SentenceTransformer, util


SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

_embedding_cache = {}
_similarity_cache = {}


def get_sbert_embedding(text):
    """
    Cache embeddings so repeated next_words do not get re-encoded constantly.
    """

    text = text.lower().strip()

    if text not in _embedding_cache:
        _embedding_cache[text] = sbert_model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

    return _embedding_cache[text]


def sbert_similarity_to_ordinate(word, ordinate=ORDINATE):
    """
    Returns cosine similarity between ORDINATE and word.
    Higher = more semantically similar to the ordinate.
    """

    word = word.lower().strip()
    ordinate = ordinate.lower().strip()

    cache_key = (ordinate, word)

    if cache_key not in _similarity_cache:
        ordinate_embedding = get_sbert_embedding(ordinate)
        word_embedding = get_sbert_embedding(word)

        score = util.cos_sim(ordinate_embedding, word_embedding).item()
        _similarity_cache[cache_key] = float(score)

    return _similarity_cache[cache_key]
    
    
    
    
    # -------------------------
# STEP B2: Extract acro events with ordinate_score
# -------------------------

from tqdm import tqdm
def extract_acro_events(
    sentences,
    dataset_kind,
    source_type,
    n,
    null_run_id=None,
    ordinate=ORDINATE
):
    rows = []

    # Tokenize first so we can know how many SBERT comparisons are coming.
    tokenized_sentences = [
        (sentence_id, tokenize(sentence))
        for sentence_id, sentence in enumerate(sentences)
    ]

    total_comparisons = sum(
        max(0, len(words) - n)
        for _, words in tokenized_sentences
    )

    progress_desc = f"Scoring {source_type} next_words vs {ordinate}"

    with tqdm(
        total=total_comparisons,
        desc=f"{source_type}→{ordinate}",
        unit="word",
        dynamic_ncols=True,
        leave=False
    ) as pbar:
        for sentence_id, words in tokenized_sentences:

            # Need n acro words plus one next word.
            for i in range(0, len(words) - n):
                acro_words = words[i:i+n]
                next_word = words[i+n]

                acro = "".join(first_initial(w) for w in acro_words)
                next_initial = first_initial(next_word)

                ordinate_score = sbert_similarity_to_ordinate(
                    next_word,
                    ordinate=ordinate
                )

                rows.append({
                    "dataset_kind": dataset_kind,
                    "source_type": source_type,
                    "null_run_id": null_run_id,
                    "sentence_id": sentence_id,
                    "word_position": i,
                    "n": n,
                    "acro": acro,
                    "next_word": next_word,
                    "next_initial": next_initial,
                    "ordinate": ordinate,
                    "ordinate_score": ordinate_score,
                })

                pbar.update(1)

    return rows
    
    
    
    
typical_events = extract_acro_events(
    sentences=typical_sentences,
    dataset_kind="typical",
    source_type="leipzig",
    n=2,
    ordinate=ORDINATE
)

null_events = extract_acro_events(
    sentences=null_sentences,
    dataset_kind="null",
    source_type="leipzig",
    n=2,
    null_run_id=1,
    ordinate=ORDINATE
)
    
    
    
    
# -------------------------
# STEP B3: Score acro → semantic ordinate
# -------------------------

def score_acro_next_ordinate_score(rows, acro, ordinate=ORDINATE):
    relevant = [
        r for r in rows
        if r["acro"] == acro and r["ordinate"] == ordinate
    ]

    if len(relevant) == 0:
        return None

    rate_ordinate_score = (
        sum(r["ordinate_score"] for r in relevant) / len(relevant)
    )

    return {
        "acro": acro,
        "ordinate": ordinate,
        "count_acro": len(relevant),
        "rate_ordinate_score": rate_ordinate_score,
    }


real_bs_happiness = score_acro_next_ordinate_score(
    typical_events,
    acro=ACRO,
    ordinate=ORDINATE
)

null_bs_happiness = score_acro_next_ordinate_score(
    null_events,
    acro=ACRO,
    ordinate=ORDINATE
)

print(f"Real {ACRO}→{ORDINATE}:", real_bs_happiness)
print(f"Null {ACRO}→{ORDINATE}:", null_bs_happiness)



from itertools import product
list26 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#list26 = ['a', 'b']
score_list = []

for length in range(1, 3):

    typical_events_by_ordinate = {}
    null_events_by_ordinate = {}

    # Preload all ordinate-specific event sets first
    for ORDINATE in ordinate_list:
        typical_events_by_ordinate[ORDINATE] = extract_acro_events(
            sentences=typical_sentences,
            dataset_kind="typical",
            source_type="leipzig",
            n=length,
            ordinate=ORDINATE
        )

        null_events_by_ordinate[ORDINATE] = extract_acro_events(
            sentences=null_sentences,
            dataset_kind="null",
            source_type="leipzig",
            n=length,
            null_run_id=1,
            ordinate=ORDINATE
        )

    # Now loop through acronyms
    for letters in product(list26, repeat=length):
        acronym = "".join(letters).upper()

        # Score this acronym against every ordinate back-to-back
        for ORDINATE in ordinate_list:
            real_score = score_acro_next_ordinate_score(
                typical_events_by_ordinate[ORDINATE],
                acro=acronym,
                ordinate=ORDINATE
            )

            null_score = score_acro_next_ordinate_score(
                null_events_by_ordinate[ORDINATE],
                acro=acronym,
                ordinate=ORDINATE
            )

            if real_score is None:
                continue

            print(f"Real {acronym}→{ORDINATE}:", real_score)

            score_list.append((
                length,
                acronym,
                ORDINATE,
                real_score["rate_ordinate_score"],
                real_score["count_acro"]
            ))

def print_score_list_multiOrdinate(score_list, sortby="acronym", reverse=False, keepPercent=0.5):
    def sortby_multiOrdinate(grouped_data, sortby="acronym", reverse=False):
        def get_sort_value(item):
            if sortby == "length":
                return item["length"]

            if sortby == "acronym":
                return item["acronym"]

            if sortby == "count":
                return item["count"]

            # Otherwise, assume sortby is an ordinate word, like "happiness"
            for ordinate, score in item["ordinates"]:
                if ordinate == sortby:
                    return score

            # If that ordinate is missing, push it to the bottom
            return float("-inf")

        return sorted(grouped_data, key=get_sort_value, reverse=reverse)

    grouped = {}

    for length, acronym, ordinate, score, count in score_list:
        key = (length, acronym)

        grouped.setdefault(key, {
            "length": length,
            "acronym": acronym,
            "count": count,
            "ordinates": []
        })

        grouped[key]["ordinates"].append((ordinate, score))

    sorted_data = list(grouped.values())

    # Keep only top half by count first
    sorted_data = sorted(sorted_data, key=lambda x: x["count"], reverse=True)
    sorted_data = sorted_data[:int(len(sorted_data) * keepPercent)]

    # Then sort by whatever you requested
    sorted_data = sortby_multiOrdinate(
        sorted_data,
        sortby=sortby,
        reverse=reverse
    )

    for item in sorted_data:
        ordinate_text = "    ".join(
            f"{ordinate.capitalize()}={score:.4f}"
            for ordinate, score in item["ordinates"]
        )

        print(f"{item['acronym']}: {ordinate_text}    Count={item['count']}")

print_score_list_multiOrdinate(
    score_list,
    sortby=ordinate_list[0],#ordinate_list[0],
    reverse=False,
    keepPercent = 0.5
)




# -------------------------
# STEP B4: Many null worlds
# -------------------------

def run_many_nulls(
    real_sentences,
    vocab,
    weights,
    n,
    acro,
    ordinate=ORDINATE,
    runs=100
):
    null_scores = []

    for run_id in range(runs):
        null_sentences = generate_null_sentences(
            real_sentences,
            vocab,
            weights
        )

        null_events = extract_acro_events(
            sentences=null_sentences,
            dataset_kind="null",
            source_type="leipzig",
            n=n,
            null_run_id=run_id,
            ordinate=ordinate
        )

        score = score_acro_next_ordinate_score(
            null_events,
            acro,
            ordinate
        )

        if score is not None:
            null_scores.append(score["rate_ordinate_score"])

    return null_scores


def z_score(real_value, null_values):
    mu = statistics.mean(null_values)
    sd = statistics.stdev(null_values)

    return {
        "real": real_value,
        "null_mean": mu,
        "null_sd": sd,
        "z": (real_value - mu) / sd if sd != 0 else None
    }

if(RUN_MANY_NULLS):
    null_bs_happiness_rates = run_many_nulls(
        real_sentences=typical_sentences,
        vocab=vocab,
        weights=weights,
        n=2,
        acro=ACRO,
        ordinate=ORDINATE,
        runs=100
    )

    real_bs_happiness_rate = real_bs_happiness["rate_ordinate_score"]

    result = z_score(real_bs_happiness_rate, null_bs_happiness_rates)

    print(result)