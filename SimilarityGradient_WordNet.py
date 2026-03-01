import nltk
# print(nltk.data.path)
from nltk.corpus import wordnet as wn
from nltk.corpus import words
import numpy as np
from sklearn.linear_model import LinearRegression
import nltk
nltk.download('words')
import matplotlib.pyplot as plt
import statistics
from gensim.models import KeyedVectors
import os
import sys
import inspect
import time
import fasttext
from decimal import Decimal
from wordfreq import top_n_list


# Cache for synsets
synset_cache = {}

def get_synsets(word):
    """Return cached synsets for a word."""
    if word not in synset_cache:
        synset_cache[word] = wn.synsets(word)
    return synset_cache[word]

def compare_words(word_a, word_b):
    """
    Compare two words using WordNet synset similarity (Wu-Palmer).
    Returns a similarity score between 0 and 1.
    """
    synsets_a = get_synsets(word_a)
    synsets_b = get_synsets(word_b)
    
    if not synsets_a or not synsets_b:
        return 0.0

    max_similarity = 0.0

    for syn_a in synsets_a:
        for syn_b in synsets_b:
            sim = syn_a.wup_similarity(syn_b)  # Wu-Palmer similarity
            if sim is not None and sim > max_similarity:
                max_similarity = sim

    return max_similarity

def compare_word_to_list(word, word_list):
    """
    Return a dictionary of similarity scores between `word` and each word in `word_list`.
    Optimized by precomputing synsets.
    """
    word_synsets = get_synsets(word)

    # Precompute synsets for the list
    word_list_synsets = {w: get_synsets(w) for w in word_list}

    scores = {}
    for other_word, other_synsets in word_list_synsets.items():
        if not word_synsets or not other_synsets:
            scores[other_word] = 0.0
            continue

        max_similarity = 0.0
        for syn_a in word_synsets:
            for syn_b in other_synsets:
                sim = syn_a.wup_similarity(syn_b)
                if sim is not None and sim > max_similarity:
                    max_similarity = sim
        scores[other_word] = max_similarity

    return scores
    
def compare_word_to_list_average_synsets(word, word_list):
    """
    Return a dictionary of similarity scores between `word` and each word in `word_list`.
    Optimized by precomputing synsets.
    """
    word_synsets = get_synsets(word)

    # Precompute synsets for the list
    word_list_synsets = {w: get_synsets(w) for w in word_list}

    scores = {}
    for other_word, other_synsets in word_list_synsets.items():
        if not word_synsets or not other_synsets:
            scores[other_word] = 0.0
            continue

        max_similarity = 0.0
        i = 0
        for syn_a in word_synsets:
            for syn_b in other_synsets:
                sim = syn_a.wup_similarity(syn_b)
                max_similarity = max_similarity + sim
                i = i + 1
        scores[other_word] = max_similarity / i

    return scores

def linear_regression_trend(sim_dict):
    """
    sim_dict: dictionary mapping words -> similarity scores
    returns slope, intercept
    """
    # Convert dict to numpy arrays
    y = np.array(list(sim_dict.values()))            # similarity scores
    X = np.arange(len(y)).reshape(-1, 1)            # use indices as x-values

    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    return slope, intercept
    
def compare_word_to_list_fit_synset(word, word_list):
    word_synsets = get_synsets(word)

    # Precompute synsets for the list
    word_list_synsets = {w: get_synsets(w) for w in word_list}

    scores = {}
    final_slope = None
    for syn_a in word_synsets:
        for other_word, other_synsets in word_list_synsets.items():
            if not word_synsets or not other_synsets:
                scores[other_word] = 0.0
                continue

            max_similarity = 0.0
            for syn_b in other_synsets:
                sim = syn_a.wup_similarity(syn_b)
                if sim is not None and sim > max_similarity:
                    max_similarity = sim
            scores[other_word] = max_similarity
        temp_slope, temp_intercept = linear_regression_trend(scores)
        if final_slope == None or final_slope > temp_slope:
            final_slope, final_intercept = temp_slope, temp_intercept
             
    return final_slope, final_intercept
    
print("#######################################################")
print("#######################################################")
print("#######################################################")
print("#######################################################")
print("#######################################################")
print("#######################################################")
def compare_word_to_list_fit_synset_average_NOT_max(word, word_list):
    pass

def find_fit_fit_slope(list, file):
    start_time = time.perf_counter()
    fit_words = []
    fit_word = "a"
    # sims = compare_word_to_list(fit_word, list)
    # slope = linear_regression_trend(sims)[0]
    slope, intercept = compare_word_to_list_fit_synset(fit_word, list)
    fit_word_slope = slope

    for word in list:
        # sims = compare_word_to_list(word, list)
        # slope = linear_regression_trend(sims)[0]
        slope, intercept = compare_word_to_list_fit_synset(word, list)
        if(fit_word_slope > slope):
            fit_word = word
            fit_word_slope = slope
        fit_words.append((word, slope))
        # write file intermittently?
        print(f"W_NET Current word = {word}    Current word slope = {slope}")
        print(f"W_NET Fit word = {fit_word}    Fit word slope = {fit_word_slope}")
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
    
def find_fit(list, file):
    start_time = time.perf_counter()
    fit_words = []
    fit_word = "a"
    sims = compare_word_to_list(fit_word, list)
    slope = linear_regression_trend(sims)[0]
    # slope, intercept = compare_word_to_list_fit_synset(fit_word, list)
    fit_word_slope = slope

    for word in list:
        sims = compare_word_to_list_average_synsets(word, list)
        slope = linear_regression_trend(sims)[0]
        # slope, intercept = compare_word_to_list_fit_synset(word, list)
        if(fit_word_slope > slope):
            fit_word = word
            fit_word_slope = slope
        fit_words.append((word, slope))
        # write file intermittently?
        print(f"W_NET Current word = {word}    Current word slope = {slope}")
        print(f"W_NET Fit word = {fit_word}    Fit word slope = {fit_word_slope}")
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
    
def generate_word_list():
    # Make sure necessary corpora are downloaded
    nltk.download('wordnet', quiet=True)
    nltk.download('words', quiet=True)

    # Get all words from WordNet and NLTK words corpus
    wordnet_words = set(wn.all_lemma_names())
    nltk_words = set(words.words())

    # Make lowercase and take intersection
    common_words = set(w.lower() for w in wordnet_words) & set(w.lower() for w in nltk_words)

    # Alphabetical sorted list
    common_words_list = sorted(common_words)
    common_words_list = common_words_list#[::250]

    # Example: print first 20 words
    print(common_words_list[:20])
    print(f"Total words in common: {len(common_words_list)}")
    print("(word_list generated)")
    print()
    print()
    
    return common_words_list#[:6000]
    
def print_compare_word_to_list(word, list, obsolete):
    sims = []
    #for w in list:
    #    sims.append(compare_words(word, w))
    #slope, intercept = linear_regression_trend_for_arrays(sims)
    sims = compare_word_to_list_average_synsets(word, list)
    slope, intercept = linear_regression_trend(sims)
    # slope, intercept = compare_word_to_list_fit_synset(word, list)
    print(word)
    print(f"Slope: {slope}, Intercept: {intercept}")
    print(f"Final Value: {intercept + slope*len(list)}")
    print()
    
def linear_regression_trend_for_arrays(similarity_scores):
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
    if(intercept == None):
        intercept = 0

    return slope, intercept
    
def slope_of_slopes_from_file(file):
    # Read the file and create a dictionary
    word_vector = {}
    with open(file, 'r') as f:
        for line in f:
            word, value = line.strip().split()
            word_vector[word] = float(value)

    # Sort alphabetically by word
    sorted_items = sorted(word_vector.items())

    # Extract only the numbers into a list
    numbers_list = [value for word, value in sorted_items]
    
    return numbers_list

def main():
    '''
    print(wn.synsets("dog"))   # should show multiple synsets
    print(wn.synsets("cat"))   # should show multiple synsets
    print(wn.synsets("car"))   # should show multiple synsets
    print(wn.synsets("happy")) # should show multiple synsets
    print(wn.synsets("joyful"))# should show multiple synsets
    
    # Example usage
    print(compare_words("dog", "cat"))      # Similar animals → score should be > 0
    print(compare_words("dog", "car"))      # Unrelated → score should be very low
    print(compare_words("happy", "joyful")) # Synonyms → score should be high
    #'''
    
    slope, intercept = linear_regression_trend_for_arrays(slope_of_slopes_from_file("wordNet_find_fit_full_average_similarity.txt"))
    print(f"Slope: {slope}  Intercept: {intercept}")
    # find_fit_fit_slope(generate_word_list(), "wordNet_find_fit_full_single_subject_synset_MAX_NOT_MIN.txt")
    find_fit(generate_word_list(), "wordNet_find_fit_full_average_similarity.txt")
    '''
    common_words = [
    "the","be","to","of","and","a","in","that","have","I","it","for","not","on","with","he","as","you","do","at",
    "this","but","his","by","from","they","we","say","her","she","or","an","will","my","one","all","would","there",
    "their","is","are","was","were","been","has","had","can","could","should","may","might","must","if","then",
    "else","when","where","why","how","what","which","who","whom","whose","because","while","before","after",
    "during","above","below","up","down","in","out","over","under","again","further","once","here","there",
    "always","never","often","sometimes","usually","really","very","quite","just","only","also","too","enough",
    "so","than","rather","almost","about","around","between","among","through","across","toward","away",
    "back","forward","left","right","near","far","early","late","now","today","tomorrow","yesterday","morning",
    "night","day","week","month","year","time","life","world","people","person","man","woman","child","children",
    "family","friend","friends","group","company","team","place","home","house","room","school","work","job",
    "money","business","government","country","state","city","community","society","system","problem","issue",
    "question","answer","idea","fact","reason","result","change","process","development","growth","history",
    "future","past","present","moment","case","point","example","kind","type","part","side","end","beginning",
    "middle","way","method","means","form","level","rate","amount","number","size","area","value","cost","price",
    "effect","cause","purpose","goal","plan","policy","rule","law","right","power","control","authority","force",
    "energy","effort","work","task","project","program","service","product","market","trade","industry",
    "technology","science","research","study","education","learning","knowledge","information","data","skill",
    "ability","experience","training","practice","theory","model","design","structure","function","role",
    "behavior","action","activity","event","situation","condition","state","status","position","role",
    "health","care","medicine","doctor","nurse","patient","hospital","treatment","therapy","disease","illness",
    "mental","physical","emotional","social","public","private","personal","individual","collective","common",
    "general","specific","special","normal","usual","typical","rare","unique","simple","complex","easy","hard",
    "difficult","possible","impossible","likely","unlikely","certain","uncertain","clear","unclear","obvious",
    "hidden","open","closed","free","busy","full","empty","available","ready","safe","dangerous","secure",
    "strong","weak","high","low","long","short","large","small","big","little","young","old","new","recent",
    "modern","traditional","local","global","national","international","economic","political","cultural",
    "religious","spiritual","moral","ethical","legal","illegal","formal","informal","official","unofficial",
    "true","false","real","fake","natural","artificial","human","animal","plant","nature","environment","earth",
    "water","air","fire","food","drink","eat","drink","sleep","wake","walk","run","sit","stand","move","travel",
    "drive","ride","fly","go","come","leave","arrive","stay","live","die","grow","build","make","create","produce",
    "use","need","want","like","love","hate","feel","think","know","believe","understand","learn","teach",
    "remember","forget","decide","choose","try","hope","wish","expect","plan","prepare","start","begin","stop",
    "end","finish","continue","change","improve","develop","increase","reduce","add","remove","include",
    "exclude","allow","prevent","help","support","protect","save","lose","win","fail","succeed","achieve",
    "reach","meet","join","leave","share","give","take","get","receive","send","bring","carry","hold","keep",
    "put","set","place","turn","open","close","push","pull","hit","cut","break","fix","repair","clean","wash",
    "build","destroy","attack","defend","fight","argue","discuss","debate","agree","disagree","accept","reject",
    "refuse","deny","confirm","prove","test","check","measure","count","compare","analyze","evaluate","judge",
    "decide","conclude","explain","describe","define","clarify","express","communicate","speak","talk","say",
    "tell","ask","answer","listen","hear","read","write","draw","paint","sing","play","watch","see","look",
    "observe","notice","focus","attention","interest","curiosity","motivation","desire","fear","anger","sadness",
    "happiness","joy","pleasure","pain","stress","relief","peace","calm","balance","order","chaos","pattern",
    "random","chance","risk","opportunity","choice","option","alternative","path","direction","journey","story",
    "narrative","meaning","purpose","value","worth","identity","self","mind","body","soul","spirit","heart",
    "thought","memory","dream","imagination","creativity","art","music","language","word","sentence","text",
    "book","paper","letter","message","email","call","signal","sign","symbol","image","picture","photo","video",
    "sound","noise","voice","silence","light","dark","color","shape","form","line","point","space","time",
    "speed","motion","force","pressure","temperature","heat","cold","energy","power","electricity","battery",
    "machine","device","tool","computer","software","hardware","internet","network","system","code","program",
    "algorithm","function","variable","list","array","dictionary","object","class","method","loop","condition",
    "error","bug","fix","update","version","release","build","deploy","run","test","debug","optimize",
    "secure","encrypt","password","account","user","admin","access","permission","role","profile","setting",
    "option","menu","button","screen","window","page","site","platform","service","app","application",
    "mobile","desktop","server","client","cloud","storage","file","folder","path","directory","disk","memory",
    "cache","process","thread","task","queue","stack","heap","log","monitor","alert","report","dashboard",
    "metric","analysis","insight","trend","pattern","forecast","prediction","model","simulation","scenario",
    "strategy","tactic","approach","framework","architecture","design","blueprint","plan","roadmap","timeline",
    "milestone","deadline","schedule","calendar","date","time","clock","timer","alarm","reminder","note",
    "list","task","todo","goal","habit","routine","ritual","practice","exercise","training","workout",
    "health","fitness","strength","endurance","flexibility","balance","coordination","nutrition","diet",
    "protein","carbohydrate","fat","vitamin","mineral","water","hydration","sleep","rest","recovery",
    "mindfulness","meditation","focus","attention","awareness","presence","reflection","insight","wisdom",
    "understanding","knowledge","truth","meaning","purpose","value","care","compassion","empathy","kindness",
    "respect","dignity","justice","fairness","equality","freedom","rights","responsibility","duty","service",
    "contribution","impact","change","progress","improvement","growth","development","evolution","transformation"]
    common_words = top_n_list("en", 3000)
    # find_fit(sorted(common_words), "wordNet_find_fit_COMMON_WORDS_3000_average_similarity.txt")
    #'''
    word_list = generate_word_list()
    model = None
    #'''
    print_compare_word_to_list("material", word_list, model)
    print_compare_word_to_list("mental", word_list, model)
    print_compare_word_to_list("science", word_list, model)
    print_compare_word_to_list("spirit", word_list, model)
    
    print_compare_word_to_list("life", word_list, model)
    print_compare_word_to_list("death", word_list, model)
    print_compare_word_to_list("fire", word_list, model)
    print_compare_word_to_list("ice", word_list, model)
    
    print_compare_word_to_list("myth", word_list, model)
    print_compare_word_to_list("storm", word_list, model)
    print_compare_word_to_list("balance", word_list, model)
    
    print_compare_word_to_list("sun", word_list, model)
    print_compare_word_to_list("moon", word_list, model)
    print_compare_word_to_list("stars", word_list, model)
    
    print_compare_word_to_list("consume", word_list, model)
    print_compare_word_to_list("eat", word_list, model)
    print_compare_word_to_list("asceticism", word_list, model)
    print_compare_word_to_list("zen", word_list, model)
    #'''
    
if __name__ == "__main__":
    main()  # This actually runs your code