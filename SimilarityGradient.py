import numpy as np
from sklearn.linear_model import LinearRegression
import nltk
nltk.download('words')
import matplotlib.pyplot as plt
from nltk.corpus import words
import statistics
from gensim.models import KeyedVectors
import os
import sys
import inspect
import time
import fasttext
from decimal import Decimal

def load_glove_vectors(glove_file_path):
    """Load GloVe vectors into a dictionary."""
    model = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0].lower()
            try:
                vector = np.array([float(x) for x in parts[1:]])
                model[word] = vector
            except ValueError:
                # skip malformed lines
                continue
    return model

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    if vec1 is None or vec2 is None:
        return 0
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm == 0:
        return 0
    return dot / norm

def compare_word_to_list(word, word_list, model):
    """Return a list of similarity scores between `word` and each word in `word_list`."""
    word = word.lower()
    if word not in model:
        raise ValueError(f"Word '{word}' not found in the model.")
    word_vec = model[word]
    
    similarities = []
    for target_word in word_list:
        target_word = target_word.lower()
        # vec = model.get(target_word)
        vec = model[target_word]
        sim = cosine_similarity(word_vec, vec) if vec is not None else 0
        similarities.append(sim)
    return similarities
    
def compare_word_to_list_for_words(word, word_list, model):
    """Return a list of similarity scores between `word` and each word in `word_list`."""
    word = word.lower()
    if word not in model:
        raise ValueError(f"Word '{word}' not found in the model.")
    word_vec = model[word]
    
    similarities = []
    for target_word in word_list:
        target_word = target_word.lower()
        # vec = model.get(target_word)
        vec = model[target_word]
        sim = cosine_similarity(word_vec, vec) if vec is not None else 0
        similarities.append((target_word, sim))
    sorted_words = [word for word, num in sorted(similarities, key=lambda x: x[1])]
    return sorted_words
    
def compare_vec_to_list(vec, word_list, model):
    similarities = []
    for target_word in word_list:
        target_word = target_word.lower()
        # vec = model.get(target_word)
        vec_large_list_word = model[target_word]
        sim = cosine_similarity(vec, vec_large_list_word) if vec is not None else 0
        similarities.append(sim)
    return similarities
    
def compare_vec_to_list_for_word(vec, word_list, model):
    word = model[word_list[0]]
    word_vec = cosine_similarity(vec, model[word_list[0]])
    for target_word in word_list:
        target_word = target_word.lower()
        # vec = model.get(target_word)
        sim = cosine_similarity(vec, model[target_word]) if vec is not None else 0
        if(sim > word_vec):
            word_vec = sim
            word = target_word
    return word

from collections import deque
def compare_vec_to_list_for_words(number_of_words_to_return, vec, word_list, model):
    return_words = {}

    for target_word in word_list:
        target_word = target_word.lower()
        sim = cosine_similarity(vec, model[target_word])

        if len(return_words) < number_of_words_to_return:
            return_words[target_word] = sim
        else:
            # find the minimum similarity in the current top words
            min_word, min_sim = min(return_words.items(), key=lambda pair: pair[1])
            # only replace if strictly higher
            if sim > min_sim:
                del return_words[min_word]
                return_words[target_word] = sim
            # ties are discarded automatically

    # Return words sorted descending by similarity
    return [word for word, _ in sorted(return_words.items(), key=lambda x: -x[1])]

from collections import Counter

def remove_all_non_duplicates(input_list):
    # Count the occurrences of each element in the list
    counts = Counter(input_list)

    # Build a new list containing only items with a count of 1
    # This keeps only unique items that never had a duplicate
    result_list = [item for item in input_list if counts[item] != 1]
    
    return result_list

def flatten(bumpy_list):
    flat = []
    for _list in bumpy_list:
        for word in _list:
            flat.append(word)
    return flat
    '''
def remove_duplicates_from_bumpy_list_1x(bumpy):
    # Flatten everything to count
    flat = [word for sublist in bumpy for word in sublist]
    counts = Counter(flat)
    
    out = []
    for sublist in bumpy:
        new_sub = [w for w in sublist if counts[w] == 1]  # keep only words that appear once globally
        out.append(new_sub)
        
    for sublist in out:
        sublist.reverse()
    
    return out
    '''
def remove_nones_from_2d(bumpy):
    return [[x for x in row if x is not None] for row in bumpy]

#def remove_duplicates_lowest_primacy(bumpy):
def remove_duplicates_from_bumpy_list_1x(bumpy):
    # Make a copy so we don’t mutate original
    result = [row[:] for row in bumpy]  

    seen = set()  # track all words globally

    for j in range(len(bumpy[0])):  # iterate columns
        # count occurrences in this column
        counts = {}
        for i in range(len(bumpy)):
            rank = bumpy[i][j]
            counts[rank] = counts.get(rank, 0) + 1

        # replace ties with None OR if we've seen the word in another column
        for i in range(len(bumpy)):
            val = bumpy[i][j]
            if counts[val] > 1 or val in seen:
                result[i][j] = None
            else:
                seen.add(val)

    return remove_nones_from_2d(result)


def compare_vec_to_list_for_words_using_letteh_list_list(number_of_words_to_return, vec, letteh_list_list, model):
    letteh_list_list_flat = flatten(letteh_list_list)
    '''
    for i in range(0, 26):
        print(f"{i+1} list to complete vec: {compare_vec_to_list_for_words(number_of_words_to_return, vec, letteh_list_list[i], model)}")
        print(f"{i+1} list to {i+1} vec: {compare_vec_to_list_for_words(number_of_words_to_return, compress_list_to_vec(letteh_list_list[i], model), letteh_list_list[i], model)}")
        print()
    print()
    print()
    print(f"Complete list to complete vec: {compare_vec_to_list_for_words(number_of_words_to_return, vec, letteh_list_list_flat, model)}")
    print()
    '''
    bumpy = []
    for i in range(0, 26):
        print(f"Complete list to {i+1} vec: {compare_vec_to_list_for_words(number_of_words_to_return, compress_list_to_vec(letteh_list_list[i], model), letteh_list_list_flat, model)}")
        bumpy.append(compare_vec_to_list_for_words(number_of_words_to_return, compress_list_to_vec(letteh_list_list[i], model), letteh_list_list_flat, model))
        print()
    bumpy = remove_duplicates_from_bumpy_list_1x(bumpy)
    print()
    print()
    for i in range(0,26):
        print(f"Complete list to {i+1} vec (no duplicates): {bumpy[i]}")
        print()
    print()

def compress_list_to_vec(small_list, model):
    total = 0
    for word in small_list:
        total += model[word]
    return total / len(small_list)

def print_compress_compare(small_list, large_list, model):
    vec = compress_list_to_vec(small_list, model)
    sims = compare_vec_to_list(vec, large_list, model)
    # sims = sims / 100
    # sims = sims * 100
    fit_value = max(sims)
    print(f"{small_list[0][0]} Best fit word : {large_list[sims.index(fit_value)]} ({fit_value} - similarity score to small list)")
    fit_value = min(sims)
    print(f"{small_list[0][0]} Worst fit word: {large_list[sims.index(fit_value)]} ({fit_value} - similarity score to small list)")
    # sims = list of floats, large_list = list of words
    
    sims_sorted = sorted(sims)
    n = len(sims_sorted)
    if n % 2 == 1:
        # odd count → true middle
        fit_value = sims_sorted[n // 2]
        # print(str(sims.index(fit_value)))
        print(f"{small_list[0][0]} Median fit word: {large_list[sims.index(fit_value)]} ({fit_value} - similarity score to small list)")
    else:
        lower = sims_sorted[(n // 2) - 1]
        upper = sims_sorted[n // 2]
        fit_value = lower   # or upper, depending on your choice
        print(f"{small_list[0][0]} Median fit word lowah: {large_list[sims.index(fit_value)]} ({fit_value} - similarity score to small list)")
        fit_value = upper   # or upper, depending on your choice
        # print(str(sims.index(fit_value)))
        print(f"{small_list[0][0]} Median fit word uppah: {large_list[sims.index(fit_value)]} ({fit_value} - similarity score to small list)")
    fit_value = find_closest_to_average(sims)
    print(f"{small_list[0][0]} Average fit word: {large_list[sims.index(fit_value)]} ({fit_value} - similarity score to small list)")
    print()
    
def find_closest_to_average(data_list):
    if not data_list:
        return None  # Handle empty list case

    list_average = sum(data_list) / len(data_list)

    closest_value = data_list[0]
    min_difference = abs(data_list[0] - list_average)

    for num in data_list:
        current_difference = abs(num - list_average)
        if current_difference < min_difference:
            min_difference = current_difference
            closest_value = num
    return closest_value
    
def alphabetical_sort(word_list):
    """Return a new list sorted alphabetically (A-Z), case-insensitive."""
    return sorted(word_list, key=lambda w: w.lower())
    
def linear_regression_trend(similarity_scores):
    """
    Fits a linear regression line to similarity scores.
    Returns slope and intercept.
    
    similarity_scores: list of floats
    """
    # Sorting similarity_scors
    similarities = sorted(similarity_scores)
    
    # X = positions (1, 2, 3, … n), reshape for sklearn
    X = np.arange(len(similarity_scores)).reshape(-1, 1)
    y = np.array(similarity_scores)

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_

    return slope, intercept
    
"""
def find_fit(list, model):
    fit_word = "a"
    sims = compare_word_to_list(list[0], list, model)
    # sims = compare_word_to_list(fit_word, list, model)
    fit_word_slope = linear_regression_trend(sims)[0]
    # fit_words = []
    for word in list:
        sims = compare_word_to_list(word, list, model)
        slope = linear_regression_trend(sims)[0]
        if(fit_word_slope < slope):
            fit_word = word
            fit_word_slope = slope
            # fit_words.append(word)
        print(f"Current word = {word}    Current word slope = {slope}")
        print(f"Fit word = {fit_word}    Fit word slope = {fit_word_slope}")
        '''
        print(f"fit_words: ", end='')
        for w in fit_words:
            print(f"{w} ", end='')
        print()
        '''
        print()
    return fit_word
"""
def find_fit(list, model, file): # was find_fit_list
    # fit_word = "communication"
    # sims = compare_word_to_list(list[0], list, model)
    # sims = compare_word_to_list(fit_word, list, model)
    # fit_word_slope = linear_regression_trend(sims)[0]
    start_time = time.perf_counter()
    fit_words = []
    fit_word = "a"
    sims = compare_word_to_list(fit_word, list, model)
    slope = linear_regression_trend(sims)[0]
    fit_word_slope = slope

    for word in list:
        sims = compare_word_to_list(word, list, model)
        slope = linear_regression_trend(sims)[0]
        if(fit_word_slope > slope):
            fit_word = word
            fit_word_slope = slope
        fit_words.append((word, slope))
        # write file intermittently?
        print(f"Current word = {word}    Current word slope = {slope}")
        print(f"Fit word = {fit_word}    Fit word slope = {fit_word_slope}")
        print()
    fit_words = sorted(fit_words, key=lambda x: x[1])
    print()
    print(f"fit_words: ")
    for w in fit_words:
        print(f"{w}, ", end='')
    print()
    # Save to a text file
    with open(file, "w", encoding="utf-8") as f:
        # f.write(f"{len(fit_words)}\n")  # header: num_words dim
        for word, vector in fit_words:
            vec_str = f"{vector:.4e}"  # convert vector to string
            f.write(f"{word} {vec_str}\n")
    print(f"saved to {file}")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.3f} seconds")
    print()
    print()
    
    return fit_words

# MOM AND DAD *********************** beginning
def compare_word_to_word(word1, word2, model):
    return cosine_similarity(model[word1], model[word2])

def average_word_to_group(word, group, model):
    average = 0.0
    for w in group:
        average = average + compare_word_to_word(word, w, model)
    return (average - 1) / (len(group) - 1)

def averages_of_word_to_group1_vs_word_to_group2(word, group1, group2, model):
    avg1 = average_word_to_group(word, group1, model)
    avg2 = average_word_to_group(word, group2, model)
    return avg1, avg2

def averages_of_group1_to_group1_vs_group1_to_group2(group1, group2, model):
    avg_1to1 = 0
    avg_1to2 = 0
    for w in group1:
        avg1, avg2 = averages_of_word_to_group1_vs_word_to_group2(w, group1, group2, model)
        avg_1to1 = avg_1to1 + avg1
        avg_1to2 = avg_1to2 + avg2
    #print("group2: " + str(group2))
    #print()
    #print("group1: " + str(group1))
    print("(avg_1to1) / len(group1): " + str((avg_1to1) / len(group1)))
    print("(avg_1to2) / len(group1): " + str((avg_1to2) / len(group1)))
    #print()
    print()
    return avg_1to1 / len(group1), avg_1to2 / len(group1)

def average_groups_to_self_vs_groups_to_universe(list_of_groups, universe, model):
    avg_group_to_self = 0
    avg_group_to_universe = 0
    for group in list_of_groups:
        avg1to1, avg1to2 = averages_of_group1_to_group1_vs_group1_to_group2(group, universe, model)
        avg_group_to_self = avg_group_to_self + avg1to1
        avg_group_to_universe = avg_group_to_universe + avg1to2
        
    #avg_group_to_self *
    return avg_group_to_self / len(list_of_groups), avg_group_to_universe / len(list_of_groups) #THIS IS WHERE I WANT TO DO THE MATH

def print_average_groups_to_self_vs_groups_to_universe(list_of_groups, universe, model):
    start_time = time.perf_counter()
    
    avg_group_to_self, avg_group_to_universe = average_groups_to_self_vs_groups_to_universe(list_of_groups, universe, model)
    print("avg_group_to_self: " + str(avg_group_to_self))
    print("avg_group_to_universe: " + str(avg_group_to_universe))
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.3f} seconds")

#MOM AND DAD *********************** end
    
#JUST DAD god and devil, good vs evil, big vs small, nice vs nasty, smart vs dumb, cat vs dog, prince vs princess, red vs black, spring vs fall, cold vs hot, run vs walk, wet vs dry
def print_compare_word_to_word_for_each_isolated_vector_element(word1, word2, vector_elements, model):
    for vector_element in vector_elements:
        print(f"{word1} vs {word2}, element {vector_element}: {compare_word_to_word_vector_element(word1, word2, vector_element, model)}")
    print()
    
def compare_word_to_word_vector_element(word1, word2, vector_element, model):
    return cosine_similarity(model[word1][vector_element], model[word2][vector_element])
#JUST DAD end

#JUST MOM Chunk and find local slopes. (Average slope across 5-50+ word chunks)
def chunk_and_average_local_slopes(chunk_size, list, model, file):
    # fit_word = "communication"
    # sims = compare_word_to_list(list[0], list, model)
    # sims = compare_word_to_list(fit_word, list, model)
    # fit_word_slope = linear_regression_trend(sims)[0]
    fit_words = []
    fit_word = "a"
    sims = compare_word_to_list(fit_word, list, model)
    slope = linear_regression_trend(sims)[0]
    fit_word_slope = slope

    for word in list:
        # print("INTEGRATE CHUNKING and")
        # print("INTEGRATE COMPARISON_LIST METHOD, OVERLOAD THE METHOD")
        sims = compare_word_to_list(word, list, model)
        slope = linear_regression_trend(sims)[0]
        if(fit_word_slope > slope):
            fit_word = word
            fit_word_slope = slope
        fit_words.append((word, slope))
        # write file intermittently?
        print(f"Current word = {word}    Current word slope = {slope}")
        print(f"Fit word = {fit_word}    Fit word slope = {fit_word_slope}")
        print()
    fit_words = sorted(fit_words, key=lambda x: x[1])
    print()
    print(f"fit_words: ")
    for w in fit_words:
        print(f"{w}, ", end='')
    print()
    # Save to a text file
    with open(file, "w", encoding="utf-8") as f:
        # f.write(f"{len(fit_words)}\n")  # header: num_words dim
        for word, vector in fit_words:
            vec_str = f"{vector:.4e}"  # convert vector to string
            f.write(f"{word} {vec_str}\n")
    print(f"saved to {file}")
    print()
    print()
    
    return fit_words
#'''
def create_comparison_list(list, model, file):
    start_time = time.perf_counter()
    fit_words = []
    fit_word = "a"
    sims = compare_word_to_list(fit_word, list, model)
    slope = linear_regression_trend(sims)[0]
    fit_word_slope = slope
    object_list = list

    for word in list:
        word = word.lower()
        sims = compare_word_to_list(word, list, model)
        print(word)
        i = 0
        for w in object_list:
            fit_words.append((word, w, sims[i]))
            i = i + 1
            # write intermittently?
        object_list = object_list[1:]
    # Save to a text file
    with open(file, "w", encoding="utf-8") as f:
        # f.write(f"{len(fit_words)}\n")  # header: num_words dim
        for word, w, vector in fit_words:
            vec_str = f"{vector}"  # convert vector to string
            f.write(f"{word} {w} {vec_str}\n")
    print(f"saved to {file}")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.3f} seconds")
    print()
    print()
    
    return fit_words #'''

def compare_word_to_list_comparison_list(word, list, model, comparison_list_file):
    # word = word.lower()
    # list = list.lower()
    if word not in list:
        raise ValueError(f"Word '{word}' not found in the list.")
    word_vec = model[word]
    w = word
    similarities = []

    with open(comparison_list_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue  # skip malformed lines

            a, b, value = parts
            if a == w or b == w:
                similarities.append(float(value))
    #USE THIS IN PLACE OF compare_word_to_list and overload find_fit type functions
    #ACTUALLY this is only a marginal speed bump best case, and it introduces code and organizational complexity that may induce errors; not necessarily negative overall, but not worth-while without aditional prompting
    #eh, I just did it
    return similarities

def find_fit_comparison_list(list, model, file, comparison_list_file): # was find_fit_list
    # fit_word = "communication"
    # sims = compare_word_to_list(list[0], list, model)
    # sims = compare_word_to_list(fit_word, list, model)
    # fit_word_slope = linear_regression_trend(sims)[0]
    start_time = time.perf_counter()
    
    fit_words = []
    fit_word = "a"
    sims = compare_word_to_list_comparison_list(fit_word, list, model, comparison_list_file)
    slope = linear_regression_trend(sims)[0]
    fit_word_slope = slope

    for word in list:
        sims = compare_word_to_list_comparison_list(word, list, model, comparison_list_file)
        slope = linear_regression_trend(sims)[0]
        if(fit_word_slope > slope):
            fit_word = word
            fit_word_slope = slope
        fit_words.append((word, slope))
        # write file intermittently?
        print(f"Current word = {word}    Current word slope = {slope}")
        print(f"Fit word = {fit_word}    Fit word slope = {fit_word_slope}")
        print()
    fit_words = sorted(fit_words, key=lambda x: x[1])
    print()
    print(f"fit_words: ")
    for w in fit_words:
        print(f"{w}, ", end='')
    print()
    # Save to a text file
    with open(file, "w", encoding="utf-8") as f:
        # f.write(f"{len(fit_words)}\n")  # header: num_words dim
        for word, vector in fit_words:
            vec_str = f"{vector:.4e}"  # convert vector to string
            f.write(f"{word} {vec_str}\n")
    print(f"saved to {file}")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.3f} seconds")
    print()
    print()
    
    return fit_words

#JUST MOM end

# JUST BEN: just compare everything to z
def just_compare_everything_to_z(everything, z, model):
    list_of_groups = everything
    universe = z
    print_average_groups_to_self_vs_groups_to_universe(list_of_groups, universe, model)

#JUST BEN end

def slopes_list(list, model):
    slopes_list = []
    for word in list:
        sims = compare_word_to_list(word, list, model)
        slope = linear_regression_trend(sims)[0]
        slopes_list.append(slope)
    return slopes_list
    
def print_slopes_list(list, model):
    slopes = []
    i = 0
    for word in list:
        sims = compare_word_to_list(word, list, model)
        slope = linear_regression_trend(sims)[0]
        slopes.append(slope)
        print(f"slope and word: {slope} {word}")
        if i % 10 == 0:
            slope, intercept = linear_regression_trend(slopes)
            print()
            print(f"Slope: {slope}, Intercept: {intercept}")
            print(f"Interim Value: {intercept + slope*len(slopes)}")
        i = i + 1
    print()
    print("____DONE___________________")
    slope, intercept = linear_regression_trend(slopes)
    print(f"Slope: {slope}, Intercept: {intercept}")
    print(f"Final Value: {intercept + slope*len(slopes)}")
    print()
    print()
    return slopes
    
# considah how I calculate horse_shoe and that it might be wrong
'''
def find_fit_horse_shoe(list, model):
    listAM = []
    listNZ = []
    i = 0
    while list[i][0].lower() != "n":
        listAM.append(list[i])
        i = i + 1
    while i < len(list):
        listNZ.append(list[i])
        i = i + 1
    
    fit_word = list[0]
    # fit_word = "associate"
    simsAM = compare_word_to_list(list[0], listAM, model)
    # sims = compare_word_to_list(fit_word, list, model)
    fit_word_slopeAM = linear_regression_trend(simsAM)[0]
    simsNZ = compare_word_to_list(list[0], listNZ, model)
    fit_word_slopeNZ = linear_regression_trend(simsNZ)[0]
    fit_word_slope = (len(listAM)*fit_word_slopeAM + len(listNZ)*fit_word_slopeNZ) / len(list)
    # fit_words = ["associate"]
    for word in list:
        simsAM = compare_word_to_list(word, listAM, model)
        simsNZ = compare_word_to_list(word, listNZ, model)
        slopeAM = linear_regression_trend(simsAM)[0]
        slopeNZ = linear_regression_trend(simsNZ)[0]
        slope = (len(listAM)*slopeAM + len(listNZ)*slopeNZ) / len(list)
        if(fit_word_slope > slope):
            fit_word = word
            fit_word_slope = slope
            # fit_words.append(word)
        print(f"Current word = {word}    Current word MID slope = {slope}")
        print(f"Fit word = {fit_word}    Fit word MID slope = {fit_word_slope}")
        # print(f"fit_words: ", end='')
        # for w in fit_words:
            # print(f"{w} ", end='')
        # print()
        print()
    return fit_word
'''
    
def print_linear_regression_trend(sims):
    slope, intercept = linear_regression_trend(sims)
    print(f"Slope: {slope}, Intercept: {intercept}")
    print(f"Final Value: {intercept + slope*len(sims)}")
    
def print_compare_word_to_list(word_to_compare, word_list, model):
    print(word_to_compare)
    sims = compare_word_to_list(word_to_compare, word_list, model)
    print_linear_regression_trend(sims)
    print_compare_word_to_list_horse_shoe(word_to_compare, word_list, model)
    print()
    
# considah how I calculate horse_shoe and that it might be wrong
def compare_word_to_list_horse_shoe(word, word_list, model):
    """Return a list of similarity scores between `word` and each word in `word_list`. [horse_shoe]"""
    listAM = []
    listNZ = []
    i = 0
    # while word_list[i][0].lower() != "n":
    while i < len(word_list) / 2:
        listAM.append(word_list[i])
        i = i + 1
    while i < len(word_list):
        listNZ.append(word_list[i])
        i = i + 1
        
    word = word.lower()
    if word not in model:
        raise ValueError(f"Word '{word}' not found in the model.")
    word_vec = model[word]
    
    similaritiesAM = []
    similaritiesNZ = []
    for target_word in listAM:
        target_word = target_word.lower()
        # vec = model.get(target_word)
        vec = model[target_word]
        sim = cosine_similarity(word_vec, vec) if vec is not None else 0
        similaritiesAM.append(sim)
    for target_word in listNZ:
        target_word = target_word.lower()
        # vec = model.get(target_word)
        vec = model[target_word]
        sim = cosine_similarity(word_vec, vec) if vec is not None else 0
        similaritiesNZ.append(sim)
        
    slopeAM, interceptAM= linear_regression_trend(similaritiesAM)
    slopeNZ, interceptNZ = linear_regression_trend(similaritiesNZ)
    fit_word_slope = (len(listAM)*slopeAM + len(listNZ)*slopeNZ) / len(word_list)
    return fit_word_slope
    
def print_compare_word_to_list_horse_shoe(word, word_list, model):
    slope = compare_word_to_list_horse_shoe(word, word_list, model)
    print(f"Current word = {word}    Current word MID slope = {slope}")
    print()
    
def average_list_slope(small_list, full_list, model):
    slopes = 0
    intercepts = 0
    i = 0
    for word in small_list:
        slopes += linear_regression_trend(compare_word_to_list(word, full_list, model))[0]
        intercepts += linear_regression_trend(compare_word_to_list(word, full_list, model))[1]
        i = i + 1
    return slopes / i, intercepts / i
    
def print_average_list_slope(list, full_list, model):
    slope, intercept = average_list_slope(list, full_list, model)
    print(list[0][0]+" average slope = "+str(slope))
    print("    average intercept = "+str(intercept)+"    final = "+str(intercept + slope * len(full_list)))
    
def word_list_glove(model):
    # Retrieve the list of English words
    english_words = set(words.words())
    
    print("length of nltk list: "+str(len(english_words)))

    # model is your loaded GloVe dictionary
    # word_list = list(model.index.keys())
    # word_list = list(model.index_to_key)
    word_list = list(model.keys())
    print("length of GloVe dictionary: "+str(len(word_list)))
    word_list = [w for w in model.keys() if w.isalpha()]
    # word_list = [w for w in model.index_to_key if w.isalpha()]
    print("length of GloVe dictionary with special characters removed: "+str(len(word_list)))
    word_list = [w for w in model.keys() if w in english_words]
    # word_list = [w for w in model.index_to_key if w in english_words]
    print("length of GloVe dictionary with special characters removed curated by nltk: "+str(len(word_list)))
    word_list = alphabetical_sort(word_list)
    # word_list = [w for w in word_list if w in model] #trash line?
    
    return word_list
    
def word_list_Word2Vec(model):
    # Retrieve the list of English words
    english_words = set(words.words())
    print("length of nltk list: "+str(len(english_words)))
    
    # 1. Start from the model’s vocabulary
    word_list = model.index_to_key
    print("model_length: "+str(len(word_list)))
    
    # 3. Keep only words in NLTK dictionary (optional)
    word_list = [w for w in word_list if w.lower() in english_words]
    print("curated by NLTK (length used in test): "+str(len(word_list)))
    
    return word_list
    
def sort_keyedvectors_word2vec(model, key_func=str.lower):
    print(model["Aardvark"])
    # Get all words sorted using the provided function
    sorted_words = sorted(model.key_to_index.keys(), key=key_func)
    sorted_words = [w for w in sorted_words if w.isalpha()]
    print(str(sorted_words.index("Aardvark")))
    
    # Create a new KeyedVectors with the same vector size
    sorted_model = KeyedVectors(vector_size=model.vector_size)

    # Prepare the vectors in sorted order
    vectors = np.array([model[word] for word in sorted_words])

    # Add them to the new model
    sorted_model.add_vectors(sorted_words, vectors)
    print(sorted_model["Aardvark"])

    return sorted_model
    
def combine_duplicates_word2vec_model(path):
    # Load your existing model
    # model = KeyedVectors.load(path, mmap='r')
    model = KeyedVectors.load_word2vec_format(path, binary=True)
    # Create a new, empty KeyedVectors with the same vector size
    cleaned = KeyedVectors(vector_size=model.vector_size)
    
    # pre clean the model
    model = sort_keyedvectors_word2vec(model)

    # Dictionary to store lowercase -> vector
    temp = {}
    
    i = 1
    j = 0
    k = 0
    standard = 0.1
    # ave_word = ""
    for word in model.key_to_index:
        lower = word.lower()
        # print(word)
        if lower not in temp:
            if(i > 1):
                temp[ave_word.lower()] = temp[ave_word.lower()] / i
                sim_to_upper = cosine_similarity(temp[ave_word.lower()], v_upper)
                if(sim_to_upper < standard):
                    print(f"'sim_to_upper' ({w_upper}) FAIL : {sim_to_upper}")
                    j += 1
            i = 1
            temp[lower] = model[word]
        else:
            i += 1
            if(i == 2):
                k += 1
            ave_word = word
            v_upper = model[word]
            w_upper = word
            temp[lower] = temp[lower] + model[word] # add
                
            # print(f"BREAK - line numbah: {inspect.currentframe().f_lineno}")#    temp[lower] {temp[lower]}, model[word] {model[word]}")
            # sys.exit()
    #'''
    print(f"YES - line numbah: {inspect.currentframe().f_lineno}")
    print(str(len(model)))
    print(f"j FAILS / len(temp) = {str(j)} / {str(len(temp))} = {str(j / len(temp))}")
    print(f"len(model) = {str(len(model))}")
    print(f"j FAILS / OPPORTUNITIES = {str(j)} / {str(k)} = {str(j / k)}")
    print(f"standard = {standard}")
    print("    a FAIL is sim_to_upper < standard")
    # sys.exit()
    #'''
    # Now add all lowercase words and vectors to new model
    cleaned.add_vectors(list(temp.keys()), list(temp.values()))

    # Save the cleaned version
    cleaned.save("model_cleaned_lowercase.model")
    return cleaned
    
def sort_list_by_back_to_front(list):
    new_list = []
    final_list = []
    for w in list:
        w = w[::-1]
        new_list.append(w)
    new_list = sorted(new_list)
    for w in new_list:
        w = w[::-1]
        final_list.append(w)
    return final_list
    
def main():
    model_numbah = 0
    sub_model_numbah = 1
    cc_wiki = 0
    if(model_numbah == 0):
        # model_name = "glove.6B.50d.txt"  # update path if needed
        # model_name = "glove.6B.300d.txt"  # update path if needed
        model_name = "glove.840B.300d.txt"  # update path if needed
        model = load_glove_vectors(model_name)
        word_list = word_list_glove(model)
        word_list = [w.lower() for w in word_list]
        word_list_back_sort = sort_list_by_back_to_front(word_list)
        # print("Original list length:", len(word_list))
        # print("Back Sorted list length:", len(word_list))
        ###remove### text_file = "GLOVE_glove.840B.300d.txt"
        
    elif(model_numbah == 1):
        # Path to your downloaded file
        path = r"C:\\Users\\jonny\\gensim-data\\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin"
        # Load the Word2Vec vectors
        model = KeyedVectors.load_word2vec_format(path, binary=True)  # Google News is binary
        model = combine_duplicates_word2vec_model(path)
        model_name = "word2vec 300 Google News"
        word_list = word_list_Word2Vec(model)
        
    elif(model_numbah == 2):
        # Path to the .vec file
        path = r"C:\Users\jonny\Downloads\Code\LetterAnalysis"
        if sub_model_numbah == 0:
            path = r"C:\Users\jonny\Downloads\Code\LetterAnalysis\cc.en.300.bin"
            model_name = "cc.en.300.bin"
        else:
            path = r"C:\Users\jonny\Downloads\Code\LetterAnalysis\cc.en.300.vec.gz"
            model_name = "cc.en.300.vec.gz"
        print("about to load model")
        
        english_words = set(words.words())
        # Load vectors
        model = fasttext.load_model(path)
        word_list = sorted([w for w in model.get_words() if w.isalpha() and w.islower() and w in english_words])
        word_list_back_sort = sort_list_by_back_to_front(word_list)
        text_file = "FAST_TEXT_cc.en.300.bin.txt"
        
    elif(model_numbah == 3):
        path = r"C:\Users\jonny\Downloads\Code\LetterAnalysis"
        path = r"C:\Users\jonny\Downloads\Code\LetterAnalysis"
        
        # Load vectors
        model = KeyedVectors.load_word2vec_format(path, binary=False)
        # model = combine_duplicates_word2vec_model(path)
        word_list = word_list_fastText(model)
        model_name = "INSERT NAME HERE"
        
    else:
        print("____BREAK_________________________")
        print("    did you set \"model_numbah\" to align with a corresponding model? Ctrl-f for \"model_numbah\"")
        print("    also considah \"sub_model_numbah\" immediately below the model_numbah variable")
        print("    basically this if statement should nevah hit \"else:\" .") 
        print("    pahforming sys.exit()")
        sys.exit()
    print(f"Number of words in test: {len(word_list)}")
    print(word_list[:50])  # first 50 words as a sample
    
    print()
    print()

    find_fit("    basically this if statement should".split(), model, "aaa_temp.txt")
    print(compare_vec_to_list_for_word(compress_list_to_vec(word_list, model), word_list, model))
    print(compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), word_list, model))
    ###remove###print(text_file)
    
    find_fit(word_list, model, "lazy.txt")
    
    '''
    print("TEST SUCCSESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # the comparison list slows things down because unless I want to make extraction much more efficient, it's searching every connection total to find all the given word connections which takes too much time
    word_list = "Racism functions as a manufactured hierarchy that extracts power through fear mythmaking dehumanization rather than biology or fate History reveals patterns where labels become weapons normalizing violence while masking economic exploitation Cultures suffer when prejudice narrows empathy fractures solidarity distorts policy education medicine art Resistance emerges via listening accountability memory repair requiring courage humility redistribution shared dignity Future justice demands dismantling structures nurturing plural belonging so every life counts fully game hell in joke kill like make note open prime quit run should talk umbrella vent walk xenon yes zip".split()
    word_list = [w.lower() for w in word_list]
    word_list = sorted(word_list)
    comparison_list_file = "COMPARISON_LIST_full.glove.840B.300d.txt"
    print(f"comparison_list_file = \"{comparison_list_file}\" ")
    comparison_list_file = "COMPARISON_LIST_temp.glove.840B.300d.txt" 
    create_comparison_list(word_list, model, comparison_list_file)
    text_file = "racism_word_list_test_text_file.glove.840B.300d.txt"
    find_fit(word_list, model, text_file)
    # YOU MUST COMMENT ONE FIND FIT FUNCTION OUT AT A TIME
    find_fit_comparison_list(word_list, model, text_file, comparison_list_file)
    print_slopes_list("    basically this if statement should".split(), model)
    # print_slopes_list(word_list, model)#'''
    '''
    #JUST DAD god and devil, good vs evil, big vs small, nice vs nasty, smart vs dumb, cat vs dog, prince vs princess, red vs black, spring vs fall, cold vs hot, run vs walk, wet vs dry
    vector_elements = [0,1,2, 50, 80, 133, 250, 275, 299]
    print_compare_word_to_word_for_each_isolated_vector_element("god", "devil", vector_elements, model)
    print_compare_word_to_word_for_each_isolated_vector_element("good", "evil", vector_elements, model)
    # ... Probably should try a-b/(a+b) instead of cosine similarity
    #JUST DAD END '''
    #'''
    #JUST MOM
    #create_comparison_list(word_list[:25], model, "COMPARISON_LIST_SMALL_TEST." + model_name)
    #create_comparison_list(word_list, model, "COMPARISON_LIST." + model_name)
    
    #JUST MOM END '''
    '''
    words = []
    with open("GLOVE_glove.840B.300d.txt") as f:
        for line in f:
            word = line.split()[0]
            words.append(word)
    #print(words)
    find_fit("    basically this if statement should basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement ".split(), model, "12_7_2025.txt")
    find_fit(words, model, "GLOVE_LEVEL_TWO.txt") 
    words = []
    with open("GLOVE_LEVEL_TWO.txt") as f:
        for line in f:
            word = line.split()[0]
            words.append(word)
    #print(words)
    find_fit("    basically this if statement should basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement ".split(), model, "12_7_2025.txt")
    find_fit(words, model, "GLOVE_LEVEL_THREE.txt")
    
    words = []
    with open("GLOVE_LEVEL_THREE.txt") as f:
        for line in f:
            word = line.split()[0]
            words.append(word)
    #print(words)
    find_fit("    basically this if statement should basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement ".split(), model, "12_7_2025.txt")
    find_fit(words, model, "GLOVE_LEVEL_FOUR.txt")

    words = []
    with open("GLOVE_LEVEL_FOUR.txt") as f:
        for line in f:
            word = line.split()[0]
            words.append(word)
    #print(words)
    find_fit("    basically this if statement should basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement ".split(), model, "12_7_2025.txt")
    find_fit(words, model, "GLOVE_LEVEL_FIVE.txt")
    
    words = []
    with open("GLOVE_LEVEL_FIVE.txt") as f:
        for line in f:
            word = line.split()[0]
            words.append(word)
    #print(words)
    find_fit("    basically this if statement should basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement ".split(), model, "12_7_2025.txt")
    find_fit(words, model, "GLOVE_LEVEL_SIX.txt")
    
    words = []
    with open("GLOVE_LEVEL_SIX.txt") as f:
        for line in f:
            word = line.split()[0]
            words.append(word)
    #print(words)
    find_fit("    basically this if statement should basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement ".split(), model, "12_7_2025.txt")
    find_fit(words, model, "GLOVE_LEVEL_SEVEN.txt")
    
    words = []
    with open("GLOVE_LEVEL_SEVEN.txt") as f:
        for line in f:
            word = line.split()[0]
            words.append(word)
    #print(words)
    find_fit("    basically this if statement should basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement ".split(), model, "12_7_2025.txt")
    find_fit(words, model, "GLOVE_LEVEL_EIGHT.txt")
    
    words = []
    with open("GLOVE_LEVEL_EIGHT.txt") as f:
        for line in f:
            word = line.split()[0]
            words.append(word)
    #print(words)
    find_fit("    basically this if statement should basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement ".split(), model, "12_7_2025.txt")
    find_fit(words, model, "GLOVE_LEVEL_NINE.txt")
    
    words = []
    with open("GLOVE_LEVEL_NINE.txt") as f:
        for line in f:
            word = line.split()[0]
            words.append(word)
    #print(words)
    find_fit("    basically this if statement should basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement basically this if statement ".split(), model, "12_7_2025.txt")
    find_fit(words, model, "GLOVE_LEVEL_TEN.txt")# '''
    
    #print("performing sys.exit()")
    #sys.exit()
    
    '''
    print_compare_word_to_list("science", word_list, model)
    print_compare_word_to_list("spirit", word_list, model)
    print_compare_word_to_list("mental", word_list, model)
    print_compare_word_to_list("material", word_list, model)
    
    print_compare_word_to_list("ascetic", word_list, model)
    print_compare_word_to_list("zen", word_list, model)
    
    print_compare_word_to_list("life", word_list, model)
    print_compare_word_to_list("death", word_list, model)
    print_compare_word_to_list("fire", word_list, model)
    print_compare_word_to_list("ice", word_list, model)
    
    print("waiting 5s")
    time.sleep(5)
    
    find_fit(word_list, model, "GLOVE_glove.840B.300d.txt")
    #find_fit(word_list, model, "FAST_TEXT_cc.en.300.bin.txt")
    #find_fit(word_list_back_sort, model, "FAST_TEXT_cc.en.300.bin_BACK_SORT.txt")
    
    #word_list_man = compare_word_to_list_for_words("man", word_list, model)
    #find_fit(word_list_back_sort, model, "GLOVE.CC.300_BACK_SORT.txt")
    #word_list_lady = compare_word_to_list_for_words("lady", word_list, model)
    #find_fit(word_list_lady, model, "find_fit_lady_GLOVE.CC.300.txt")
    '''
    # find_fit(word_list)
    # find_fit_list(word_list, model)
    
    
    print(model_name)
    
    listA = []
    listB = []
    listC = []
    listD = []
    listE = []
    listF = []
    listG = []
    listH = []
    listI = []
    listJ = []
    listK = []
    listL = []
    listM = []
    listN = []
    listO = []
    listP = []
    listQ = []
    listR = []
    listS = []
    listT = []
    listU = []
    listV = []
    listW = []
    listX = []
    listY = []
    listZ = []
    i = 0
    while word_list[i][0].lower() != "b":
        listA.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "c":
        listB.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "d":
        listC.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "e":
        listD.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "f":
        listE.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "g":
        listF.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "h":
        listG.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "i":
        listH.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "j":
        listI.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "k":
        listJ.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "l":
        listK.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "m":
        listL.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "n":
        listM.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "o":
        listN.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "p":
        listO.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "q":
        listP.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "r":
        listQ.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "s":
        listR.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "t":
        listS.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "u":
        listT.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "v":
        listU.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "w":
        listV.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "x":
        listW.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "y":
        listX.append(word_list[i])
        i = i + 1
    while word_list[i][0].lower() != "z":
        listY.append(word_list[i])
        i = i + 1
    while i != len(word_list):
        listZ.append(word_list[i])
        i = i + 1
        
    letteh_list_list = []
    letteh_list_list.append(listA)
    letteh_list_list.append(listB)
    letteh_list_list.append(listC)
    letteh_list_list.append(listD)
    letteh_list_list.append(listE)
    letteh_list_list.append(listF)
    letteh_list_list.append(listG)
    letteh_list_list.append(listH)
    letteh_list_list.append(listI)
    letteh_list_list.append(listJ)
    letteh_list_list.append(listK)
    letteh_list_list.append(listL)
    letteh_list_list.append(listM)
    letteh_list_list.append(listN)
    letteh_list_list.append(listO)
    letteh_list_list.append(listP)
    letteh_list_list.append(listQ)
    letteh_list_list.append(listR)
    letteh_list_list.append(listS)
    letteh_list_list.append(listT)
    letteh_list_list.append(listU)
    letteh_list_list.append(listV)
    letteh_list_list.append(listW)
    letteh_list_list.append(listX)
    letteh_list_list.append(listY)
    letteh_list_list.append(listZ)
    print()
    print()
    compare_vec_to_list_for_words_using_letteh_list_list(250, compress_list_to_vec(word_list, model), letteh_list_list, model)
    
    print()
    print()
    print()
    '''
    print(f"A: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listA, model)}")
    print(f"B: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listB, model)}")
    print(f"C: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listC, model)}")
    print(f"D: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listD, model)}")
    print(f"E: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listE, model)}")
    print(f"F: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listF, model)}")
    print(f"G: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listG, model)}")
    print(f"H: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listH, model)}")
    print(f"I: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listI, model)}")
    print(f"J: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listJ, model)}")
    print(f"K: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listK, model)}")
    print(f"L: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listL, model)}")
    print(f"M: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listM, model)}")
    print(f"N: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listN, model)}")
    print(f"O: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listO, model)}")
    print(f"P: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listP, model)}")
    print(f"Q: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listQ, model)}")
    print(f"R: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listR, model)}")
    print(f"S: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listS, model)}")
    print(f"T: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listT, model)}")
    print(f"U: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listU, model)}")
    print(f"V: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listV, model)}")
    print(f"W: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listW, model)}")
    print(f"X: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listX, model)}")
    print(f"Y: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listY, model)}")
    print(f"Z: {compare_vec_to_list_for_words(10, compress_list_to_vec(word_list, model), listZ, model)}")
    '''
    
    # just_compare_everything_to_z(letteh_list_list, listZ, model)
    '''
    flat = [item for sublist in letteh_list_list for item in sublist]
    # print(flat)
    
    temp = []
    for i in range(26):
        jlist = []
        for j in range(2):
            jlist.append(letteh_list_list[i][j])
        temp.append(jlist)
    letteh_list_list = temp
    flat = [item for sublist in letteh_list_list for item in sublist]
    print(flat)
    '''
    # print_average_groups_to_self_vs_groups_to_universe(letteh_list_list, flat, model)
    0.030174423595102733
    0.12552642196553085
    # print(str((compare_word_to_word('a', 'aa', model) + compare_word_to_word('aa', 'a', model))/ 2))
    # numbers = [compare_word_to_word('a', 'aa', model), compare_word_to_word('a', 'b', model), compare_word_to_word('a', 'ba', model), compare_word_to_word('a', 'c', model), compare_word_to_word('a', 'ca', model), compare_word_to_word('a', 'd', model), compare_word_to_word('a', 'da', model), compare_word_to_word('a', 'e', model), compare_word_to_word('a', 'ea', model), compare_word_to_word('a', 'f', model), compare_word_to_word('a', 'fa', model), compare_word_to_word('a', 'g', model), compare_word_to_word('a', 'ga', model), compare_word_to_word('a', 'h', model), compare_word_to_word('a', 'ha', model), compare_word_to_word('a', 'i', model), compare_word_to_word('a', 'iamb', model), compare_word_to_word('a', 'j', model), compare_word_to_word('a', 'jab', model), compare_word_to_word('a', 'k', model), compare_word_to_word('a', 'ka', model), compare_word_to_word('a', 'l', model), compare_word_to_word('a', 'la', model), compare_word_to_word('a', 'm', model), compare_word_to_word('a', 'ma', model), compare_word_to_word('a', 'n', model), compare_word_to_word('a', 'na', model), compare_word_to_word('a', 'o', model), compare_word_to_word('a', 'oaf', model), compare_word_to_word('a', 'p', model), compare_word_to_word('a', 'pa', model), compare_word_to_word('a', 'q', model), compare_word_to_word('a', 'qasida', model), compare_word_to_word('a', 'r', model), compare_word_to_word('a', 'ra', model), compare_word_to_word('a', 's', model), compare_word_to_word('a', 'sa', model), compare_word_to_word('a', 't', model), compare_word_to_word('a', 'ta', model), compare_word_to_word('a', 'u', model), compare_word_to_word('a', 'uang', model), compare_word_to_word('a', 'v', model), compare_word_to_word('a', 'vacancy', model), compare_word_to_word('a', 'w', model), compare_word_to_word('a', 'wa', model), compare_word_to_word('a', 'x', model), compare_word_to_word('a', 'xanthate', model), compare_word_to_word('a', 'y', model), compare_word_to_word('a', 'ya', model), compare_word_to_word('a', 'z', model), compare_word_to_word('a', 'za', model), compare_word_to_word('aa', 'a', model), compare_word_to_word('aa', 'b', model), compare_word_to_word('aa', 'ba', model), compare_word_to_word('aa', 'c', model), compare_word_to_word('aa', 'ca', model), compare_word_to_word('aa', 'd', model), compare_word_to_word('aa', 'da', model), compare_word_to_word('aa', 'e', model), compare_word_to_word('aa', 'ea', model), compare_word_to_word('aa', 'f', model), compare_word_to_word('aa', 'fa', model), compare_word_to_word('aa', 'g', model), compare_word_to_word('aa', 'ga', model), compare_word_to_word('aa', 'h', model), compare_word_to_word('aa', 'ha', model), compare_word_to_word('aa', 'i', model), compare_word_to_word('aa', 'iamb', model), compare_word_to_word('aa', 'j', model), compare_word_to_word('aa', 'jab', model), compare_word_to_word('aa', 'k', model), compare_word_to_word('aa', 'ka', model), compare_word_to_word('aa', 'l', model), compare_word_to_word('aa', 'la', model), compare_word_to_word('aa', 'm', model), compare_word_to_word('aa', 'ma', model), compare_word_to_word('aa', 'n', model), compare_word_to_word('aa', 'na', model), compare_word_to_word('aa', 'o', model), compare_word_to_word('aa', 'oaf', model), compare_word_to_word('aa', 'p', model), compare_word_to_word('aa', 'pa', model), compare_word_to_word('aa', 'q', model), compare_word_to_word('aa', 'qasida', model), compare_word_to_word('aa', 'r', model), compare_word_to_word('aa', 'ra', model), compare_word_to_word('aa', 's', model), compare_word_to_word('aa', 'sa', model), compare_word_to_word('aa', 't', model), compare_word_to_word('aa', 'ta', model), compare_word_to_word('aa', 'u', model), compare_word_to_word('aa', 'uang', model), compare_word_to_word('aa', 'v', model), compare_word_to_word('aa', 'vacancy', model), compare_word_to_word('aa', 'w', model), compare_word_to_word('aa', 'wa', model), compare_word_to_word('aa', 'x', model), compare_word_to_word('aa', 'xanthate', model), compare_word_to_word('aa', 'y', model), compare_word_to_word('aa', 'ya', model), compare_word_to_word('aa', 'z', model), compare_word_to_word('aa', 'za', model)]
    # total = sum(Decimal(str(x)) for x in numbers) / 102
    # print(total)
    # print(str((compare_word_to_word('a', 'aa', model) + compare_word_to_word('a', 'b', model) + compare_word_to_word('a', 'ba', model) + compare_word_to_word('a', 'c', model) + compare_word_to_word('a', 'ca', model) + compare_word_to_word('a', 'd', model) + compare_word_to_word('a', 'da', model) + compare_word_to_word('a', 'e', model) + compare_word_to_word('a', 'ea', model) + compare_word_to_word('a', 'f', model) + compare_word_to_word('a', 'fa', model) + compare_word_to_word('a', 'g', model) + compare_word_to_word('a', 'ga', model) + compare_word_to_word('a', 'h', model) + compare_word_to_word('a', 'ha', model) + compare_word_to_word('a', 'i', model) + compare_word_to_word('a', 'iamb', model) + compare_word_to_word('a', 'j', model) + compare_word_to_word('a', 'jab', model) + compare_word_to_word('a', 'k', model) + compare_word_to_word('a', 'ka', model) + compare_word_to_word('a', 'l', model) + compare_word_to_word('a', 'la', model) + compare_word_to_word('a', 'm', model) + compare_word_to_word('a', 'ma', model) + compare_word_to_word('a', 'n', model) + compare_word_to_word('a', 'na', model) + compare_word_to_word('a', 'o', model) + compare_word_to_word('a', 'oaf', model) + compare_word_to_word('a', 'p', model) + compare_word_to_word('a', 'pa', model) + compare_word_to_word('a', 'q', model) + compare_word_to_word('a', 'qasida', model) + compare_word_to_word('a', 'r', model) + compare_word_to_word('a', 'ra', model) + compare_word_to_word('a', 's', model) + compare_word_to_word('a', 'sa', model) + compare_word_to_word('a', 't', model) + compare_word_to_word('a', 'ta', model) + compare_word_to_word('a', 'u', model) + compare_word_to_word('a', 'uang', model) + compare_word_to_word('a', 'v', model) + compare_word_to_word('a', 'vacancy', model) + compare_word_to_word('a', 'w', model) + compare_word_to_word('a', 'wa', model) + compare_word_to_word('a', 'x', model) + compare_word_to_word('a', 'xanthate', model) + compare_word_to_word('a', 'y', model) + compare_word_to_word('a', 'ya', model) + compare_word_to_word('a', 'z', model) + compare_word_to_word('a', 'za', model) + compare_word_to_word('aa', 'a', model) + compare_word_to_word('aa', 'b', model) + compare_word_to_word('aa', 'ba', model) + compare_word_to_word('aa', 'c', model) + compare_word_to_word('aa', 'ca', model) + compare_word_to_word('aa', 'd', model) + compare_word_to_word('aa', 'da', model) + compare_word_to_word('aa', 'e', model) + compare_word_to_word('aa', 'ea', model) + compare_word_to_word('aa', 'f', model) + compare_word_to_word('aa', 'fa', model) + compare_word_to_word('aa', 'g', model) + compare_word_to_word('aa', 'ga', model) + compare_word_to_word('aa', 'h', model) + compare_word_to_word('aa', 'ha', model) + compare_word_to_word('aa', 'i', model) + compare_word_to_word('aa', 'iamb', model) + compare_word_to_word('aa', 'j', model) + compare_word_to_word('aa', 'jab', model) + compare_word_to_word('aa', 'k', model) + compare_word_to_word('aa', 'ka', model) + compare_word_to_word('aa', 'l', model) + compare_word_to_word('aa', 'la', model) + compare_word_to_word('aa', 'm', model) + compare_word_to_word('aa', 'ma', model) + compare_word_to_word('aa', 'n', model) + compare_word_to_word('aa', 'na', model) + compare_word_to_word('aa', 'o', model) + compare_word_to_word('aa', 'oaf', model) + compare_word_to_word('aa', 'p', model) + compare_word_to_word('aa', 'pa', model) + compare_word_to_word('aa', 'q', model) + compare_word_to_word('aa', 'qasida', model) + compare_word_to_word('aa', 'r', model) + compare_word_to_word('aa', 'ra', model) + compare_word_to_word('aa', 's', model) + compare_word_to_word('aa', 'sa', model) + compare_word_to_word('aa', 't', model) + compare_word_to_word('aa', 'ta', model) + compare_word_to_word('aa', 'u', model) + compare_word_to_word('aa', 'uang', model) + compare_word_to_word('aa', 'v', model) + compare_word_to_word('aa', 'vacancy', model) + compare_word_to_word('aa', 'w', model) + compare_word_to_word('aa', 'wa', model) + compare_word_to_word('aa', 'x', model) + compare_word_to_word('aa', 'xanthate', model) + compare_word_to_word('aa', 'y', model) + compare_word_to_word('aa', 'ya', model) + compare_word_to_word('aa', 'z', model) + compare_word_to_word('aa', 'za', model)) / 102))
    
    '''
    for list in letteh_list_list:
        print(f"{list[0][0][0].capitalize()} compare_word_to_list:")
        print_compare_word_to_list("masculine", list, model)
        print_compare_word_to_list("feminine", list, model)
        print()
    '''
    """
    print("[\"word\",\"can\",\"cat\"]")
    print_average_list_slope(["word","can","cat"], word_list, model)
    print()
    
    print_compress_compare(listA, word_list, model)
    print_compress_compare(listB, word_list, model)
    print_compress_compare(listC, word_list, model)
    print_compress_compare(listD, word_list, model)
    print_compress_compare(listE, word_list, model)
    print_compress_compare(listF, word_list, model)
    print_compress_compare(listG, word_list, model)
    print_compress_compare(listH, word_list, model)
    print_compress_compare(listI, word_list, model)
    print_compress_compare(listJ, word_list, model)
    print_compress_compare(listK, word_list, model)
    print_compress_compare(listL, word_list, model)
    print_compress_compare(listM, word_list, model)
    print_compress_compare(listN, word_list, model)
    print_compress_compare(listO, word_list, model)
    print_compress_compare(listP, word_list, model)
    print_compress_compare(listQ, word_list, model)
    print_compress_compare(listR, word_list, model)
    print_compress_compare(listS, word_list, model)
    print_compress_compare(listT, word_list, model)
    print_compress_compare(listU, word_list, model)
    print_compress_compare(listV, word_list, model)
    print_compress_compare(listW, word_list, model)
    print_compress_compare(listX, word_list, model)
    print_compress_compare(listY, word_list, model)
    print_compress_compare(listZ, word_list, model)
    print()
    print_compress_compare(listA, listA, model)
    print_compress_compare(listB, listB, model)
    print_compress_compare(listC, listC, model)
    print_compress_compare(listD, listD, model)
    print_compress_compare(listE, listE, model)
    print_compress_compare(listF, listF, model)
    print_compress_compare(listG, listG, model)
    print_compress_compare(listH, listH, model)
    print_compress_compare(listI, listI, model)
    print_compress_compare(listJ, listJ, model)
    print_compress_compare(listK, listK, model)
    print_compress_compare(listL, listL, model)
    print_compress_compare(listM, listM, model)
    print_compress_compare(listN, listN, model)
    print_compress_compare(listO, listO, model)
    print_compress_compare(listP, listP, model)
    print_compress_compare(listQ, listQ, model)
    print_compress_compare(listR, listR, model)
    print_compress_compare(listS, listS, model)
    print_compress_compare(listT, listT, model)
    print_compress_compare(listU, listU, model)
    print_compress_compare(listV, listV, model)
    print_compress_compare(listW, listW, model)
    print_compress_compare(listX, listX, model)
    print_compress_compare(listY, listY, model)
    print_compress_compare(listZ, listZ, model)
    print()
    #'''
    # list26 = [listA[0][0].lower, listB[0][0].lower, listC[0][0].lower, listD[0][0].lower, listE[0][0].lower, listF[0][0].lower, listG[0][0].lower, listH[0][0].lower, listI[0][0].lower, listJ[0][0].lower, listK[0][0].lower, listL[0][0].lower, listM[0][0].lower, listN[0][0].lower, listO[0][0].lower, listP[0][0].lower, listQ[0][0].lower, listR[0][0].lower, listS[0][0].lower, listT[0][0].lower, listU[0][0].lower, listV[0][0].lower, listW[0][0].lower, listX[0][0].lower, listY[0][0].lower, listZ[0][0].lower]
    list26 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    print_compress_compare(listA, list26, model)
    print_compress_compare(listB, list26, model)
    print_compress_compare(listC, list26, model)
    print_compress_compare(listD, list26, model)
    print_compress_compare(listE, list26, model)
    print_compress_compare(listF, list26, model)
    print_compress_compare(listG, list26, model)
    print_compress_compare(listH, list26, model)
    print_compress_compare(listI, list26, model)
    print_compress_compare(listJ, list26, model)
    print_compress_compare(listK, list26, model)
    print_compress_compare(listL, list26, model)
    print_compress_compare(listM, list26, model)
    print_compress_compare(listN, list26, model)
    print_compress_compare(listO, list26, model)
    print_compress_compare(listP, list26, model)
    print_compress_compare(listQ, list26, model)
    print_compress_compare(listR, list26, model)
    print_compress_compare(listS, list26, model)
    print_compress_compare(listT, list26, model)
    print_compress_compare(listU, list26, model)
    print_compress_compare(listV, list26, model)
    print_compress_compare(listW, list26, model)
    print_compress_compare(listX, list26, model)
    print_compress_compare(listY, list26, model)
    print_compress_compare(listZ, list26, model)
    print(model_name)
    #'''
    print(f"[{len(listA)},{len(listB)},{len(listC)},{len(listD)},{len(listE)},{len(listF)},{len(listG)},{len(listH)},{len(listI)},{len(listJ)},{len(listK)},{len(listL)},{len(listM)},{len(listN)},{len(listO)},{len(listP)},{len(listQ)},{len(listR)},{len(listS)},{len(listT)},{len(listU)},{len(listV)},{len(listW)},{len(listX)},{len(listY)},{len(listZ)}]")
    # find_fit_horse_shoe(word_list)
    # find_fit(word_list, model)
    """
    '''
    print("average_list_slopes A-Z")
    print_average_list_slope(listA, word_list, model)
    print_average_list_slope(listB, word_list, model)
    print_average_list_slope(listC, word_list, model)
    print_average_list_slope(listD, word_list, model)
    print_average_list_slope(listE, word_list, model)
    print_average_list_slope(listF, word_list, model)
    print_average_list_slope(listG, word_list, model)
    print_average_list_slope(listH, word_list, model)
    print_average_list_slope(listI, word_list, model)
    print_average_list_slope(listJ, word_list, model)
    print_average_list_slope(listK, word_list, model)
    print_average_list_slope(listL, word_list, model)
    print_average_list_slope(listM, word_list, model)
    print_average_list_slope(listN, word_list, model)
    print_average_list_slope(listO, word_list, model)
    print_average_list_slope(listP, word_list, model)
    print_average_list_slope(listQ, word_list, model)
    print_average_list_slope(listR, word_list, model)
    print_average_list_slope(listS, word_list, model)
    print_average_list_slope(listT, word_list, model)
    print_average_list_slope(listU, word_list, model)
    print_average_list_slope(listV, word_list, model)
    print_average_list_slope(listW, word_list, model)
    print_average_list_slope(listX, word_list, model)
    print_average_list_slope(listY, word_list, model)
    print_average_list_slope(listZ, word_list, model)
    print()
    '''
    print("----------------Slopes list data-----------------------FIX TO PULL FROM FILE NOT REGENERATE DATA")
    sys.exit()
    slopes = slopes_list(word_list, model)
    # early alphabet fit words from Keep Notes (find fit for "associate" negative slope *4/5):
    # slopes = [-8.242044647198697e-07, -6.74846851946739e-07, -8.242044647198697e-07, -7.174670273549874e-07, -6.976666095026684e-07, -6.789112085329385e-07, -7.218036190437514e-07, -6.785909300929131e-07, -7.046237025979342e-07, -7.228708765377233e-07, -6.766645795797035e-07, -7.044955842717168e-07, -7.02059090191836e-07, -6.892233678070942e-07, -6.86721188409972e-07, -7.77592050931229e-07, -6.909569559159972e-07, -6.771787096261509e-07, -6.710145852164535e-07, -7.719022446585217e-07, -8.129446502364452e-07, -6.978682002666636e-07, -6.729354517437756e-07, -6.803844234119588e-07, -6.602802512387119e-07, -6.981244707566427e-07, -6.621481381433055e-07, -6.720372022065572e-07, -6.915308173756062e-07]
    slope, intercept = linear_regression_trend(slopes)
    print(f"Slope: {slope}, Intercept: {intercept}")
    print(f"Final Value: {intercept + slope*len(slopes)}")
    
    # Simple scatter plot
    plt.scatter(range(len(slopes)), slopes, color='blue', s=1)
    plt.title("Scatter Plot of Slopes")
    plt.xlabel("Index")
    plt.ylabel("Slope Value")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
if __name__ == "__main__":
    main()
