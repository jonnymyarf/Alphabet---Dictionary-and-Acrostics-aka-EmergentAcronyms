import nltk
import numpy as np

def jump_extractor(word):
    """
    Compute the number of forward (towards 'z') versus backward (towards 'a') 
    alphabetic jumps in the input word. Returns a string ratio "z_count:a_count".
    """
    word = word.lower()
    z_count, a_count = 0, 0
    print(f"Processing word: {word}")
    for i in range(len(word)-1):
        curr, next_char = word[i], word[i+1]
        if not curr.isalpha() or not next_char.isalpha():
            continue  # skip non-letter characters
        # Compare alphabetic order: positive -> forward (toward 'z'), negative -> backward (toward 'a')
        diff = ord(next_char) - ord(curr)
        if diff > 0:
            z_count += 1
            print(f"  {curr}->{next_char}: forward (z direction)")
        elif diff < 0:
            a_count += 1
            print(f"  {curr}->{next_char}: backward (a direction)")
        else:
            print(f"  {curr}->{next_char}: no change")
    ratio = f"{z_count}:{a_count}"
    print(f"Resulting jump ratio: {ratio}")
    return ratio

# Example usage:
jump_extractor("correct")   # Expected output logs for c->o, o->r, etc., and return "3:2"
jump_extractor("real")      # Example check for "real" (r->e (a), e->a (a), a->l (z))

def make_word_list(glove_model):
    """
    Make a list of words that appear both in NLTK's English word list and in the GloVe model.
    """
    print("Building overlapping word list using NLTK and GloVe...")
    # Ensure NLTK word corpus is downloaded
    nltk.download('words', quiet=True)
    from nltk.corpus import words
    nltk_words = set(w.lower() for w in words.words('en'))
    print(f"NLTK corpus contains {len(nltk_words)} words.")

    # Get GloVe vocabulary (assuming glove_model has .key_to_index)
    glove_vocab = set(glove_model.key_to_index.keys())
    print(f"GloVe model vocabulary size: {len(glove_vocab)} words.")
    
    overlap = sorted(nltk_words.intersection(glove_vocab))
    print(f"Number of overlapping words: {len(overlap)}")
    # For demonstration, print first few words
    print("Sample overlapping words:", overlap[:10])
    return overlap

# Usage (after loading GloVe model):
# word_list = make_word_list(glove_model)

def sort_by_jump(word_list):
    """
    Group words by their jump ratio (forward:backward). 
    Returns a dictionary like {"3:2": ["correct", ...], "1:3": ["real", ...], ...}.
    """
    sorted_by_jump = {}
    print("\nGrouping words by jump ratio...")
    for w in word_list:
        ratio = jump_extractor(w)
        sorted_by_jump.setdefault(ratio, []).append(w)
    # Print summary of groups
    for ratio, words in sorted_by_jump.items():
        print(f"Ratio {ratio}: {len(words)} words (e.g., {words[:3]})")
    return sorted_by_jump

# Usage:
# sorted_by_jump = sort_by_jump(word_list)

from gensim.models import KeyedVectors

# Load GloVe (Common Crawl 840B, 300d) – requires downloading the file manually.
# The file path 'glove.840B.300d.txt' should point to the downloaded text file.
print("\nLoading GloVe embeddings (this may take a while)...")
glove_model = KeyedVectors.load_word2vec_format('glove.840B.300d.txt', binary=False, no_header=True)
print("GloVe embeddings loaded successfully.")

def compute_group_averages(sorted_by_jump, glove_model):
    """
    For each jump-ratio key, compute the average GloVe vector of all words in that group.
    Returns a dict: {ratio: average_vector}.
    """
    sorted_by_jump_vectors = {}
    print("\nComputing average vectors for each jump-ratio group...")
    for ratio, words in sorted_by_jump.items():
        vectors = []
        for w in words:
            if w in glove_model:
                vectors.append(glove_model.get_vector(w))
            else:
                print(f"  (Warning: word '{w}' not in GloVe vocab, skipped)")
        if vectors:
            avg_vec = np.mean(vectors, axis=0)
            sorted_by_jump_vectors[ratio] = avg_vec
            print(f"  Ratio {ratio}: averaged {len(vectors)} vectors.")
        else:
            sorted_by_jump_vectors[ratio] = None
            print(f"  Ratio {ratio}: no vectors found (all words missing).")
    return sorted_by_jump_vectors

# Usage:
# sorted_by_jump_vectors = compute_group_averages(sorted_by_jump, glove_model)

def family(word, sorted_by_jump, glove_model):
    """
    Compute the average GloVe vector of all other words in the same jump-ratio group as `word`.
    """
    ratio = jump_extractor(word)
    group_words = sorted_by_jump.get(ratio, []).copy()
    if word in group_words:
        group_words.remove(word)
    print(f"\nComputing family vector for '{word}', ratio {ratio}:")
    if not group_words:
        print(f"  No other words in group {ratio}, returning zero vector.")
        return np.zeros(glove_model.vector_size)
    # Get vectors for remaining words
    vectors = [glove_model.get_vector(w) for w in group_words if w in glove_model]
    if not vectors:
        print(f"  None of the remaining words found in GloVe vocab, returning zero vector.")
        return np.zeros(glove_model.vector_size)
    avg_vec = np.mean(vectors, axis=0)
    print(f"  Averaged {len(vectors)} words: {group_words}")
    return avg_vec

# Example (after sorted_by_jump is built):
# fam_vec = family("real", sorted_by_jump, glove_model)

def meta_compare(word, sorted_by_jump, sorted_by_jump_vectors, glove_model):
    """
    Compare the family vector of `word` to the global average vector of all other groups.
    Returns the difference vector.
    """
    ratio = jump_extractor(word)
    family_vec = family(word, sorted_by_jump, glove_model)
    # Compute the average of all group-average vectors except this ratio
    other_vecs = [v for r,v in sorted_by_jump_vectors.items() if r != ratio and v is not None]
    if not other_vecs:
        print(f"No other group vectors to compare with for ratio {ratio}.")
        return np.zeros_like(family_vec)
    global_avg = np.mean(other_vecs, axis=0)
    if word in glove_model:
        word_vec = glove_model.get_vector(word)
    else:
        print(f"  [meta_compare] word '{word}' missing in GloVe -> using zero vector")
        word_vec = np.zeros(glove_model.vector_size)
    diff = cosine_sim(family_vec, word_vec) - cosine_sim(global_avg, word_vec)
    print(f"Meta-compare for '{word}' (ratio {ratio}): computed difference vector.")
    return diff

# Example (after sorted_by_jump_vectors is built):
# diff = meta_compare("real", sorted_by_jump, sorted_by_jump_vectors, glove_model)

def cosine_sim(u, v):
    """Safe cosine similarity between two vectors (returns 0 for zero-vector cases)."""
    if u is None or v is None:
        return 0.0
    num = np.dot(u, v)
    denom = (np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0:
        return 0.0
    return float(num / denom)

if __name__ == "__main__":
    # Assume GloVe model is already loaded as glove_model
    word_list = make_word_list(glove_model)
    sorted_by_jump = sort_by_jump(word_list)
    sorted_by_jump_vectors = compute_group_averages(sorted_by_jump, glove_model)
    
    total_diff = 0
    count = 0
    print("\nRunning meta-compare on all words...")
    for w in word_list:
        diff = meta_compare(w, sorted_by_jump, sorted_by_jump_vectors, glove_model)
        total_diff += diff
        count += 1
    
    if count > 0:
        avg_diff = total_diff / count
        print("\nAverage difference vector (total_diff / count):")
        print(avg_diff)
    else:
        print("No words to process.")