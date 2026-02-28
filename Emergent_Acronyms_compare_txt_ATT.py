from collections import defaultdict, Counter
import re
from itertools import product
import sys

import requests
import os

import gzip
from io import BytesIO
import requests

#file = "OS_MATH.txt"
#file = "OS_PHYSICS.txt"
#file = "FHSST_MATH.txt"
#file = "FHSST_PHYSICS.txt"
#file = "LOGIC.txt"
#file2 = "CAPITAL.txt"
#file2 = "KANT.txt"


#file = "ORWELL_SHORT.txt"
#file2 = "HOBBES_SHORT.txt"


file = "Shakespere.txt"
file2 = "Zizek_SOI.txt"
file = "Hegel.txt"

file = "ORWELL.txt"
file2 = "HOBBES.txt"

file = "PLATO.txt"
file2 = "ARISTOTLE.txt"

#file = "Torah.txt"
#file2 = "Quran.txt"

file2 = "Physicalists.txt"
file = "Idealists.txt"

file = "Philosophy_ArchiveTxt_ROWS_2_PAGES_200_START_PAGE_1.txt"
file2 = "Philosophy_ArchiveTxt_ROWS_2_PAGES_200_START_PAGE_201.txt"

file = "Philosophy_ArchiveTxt.txt"
file2 = "Short stories_ArchiveTxt.txt"

file_name = file[:-4]
file_name2 = file2[:-4]

# Step 1: Clean text
def clean_text(text):
    print("in")
    words = re.findall(r"[A-Za-z]+'?[A-Za-z]+|[A-Za-z]+", text)
    words = [w.replace("'", "").upper() for w in words]
    exclude_words = ["a", "an", "the", "and", "but", "if", "nor", "or", "so", "yet", "as", "at", "by", "for", "in", "of", "off", "on", "per", "to", "up", "via"]
    exclude_words = []
    exclude_words = [w.upper() for w in exclude_words]
    words = [w for w in words if w not in exclude_words]
    print("out")
    return words


text_lines = []
with open(file, "rb") as f:  # open in binary mode
    for raw_line in f:
        try:
            line = raw_line.decode("utf-8")  # try decoding
            text_lines.append(line)
        except UnicodeDecodeError:
            # skip this line
            continue
text = " ".join(text_lines)

text2_lines = []
with open(file2, "rb") as f:  # open in binary mode
    for raw_line in f:
        try:
            line = raw_line.decode("utf-8")  # try decoding
            text2_lines.append(line)
        except UnicodeDecodeError:
            # skip this line
            continue
text2 = " ".join(text2_lines)


#text = "This is why your old code “didn’t have this problem” — it only looked at first letters, avoiding the explosion. The moment you try to generalize subwords across multiple words, the complexity is inherently exponential in n and word lengths."
#text2 = "Reality check: There’s no way to make this truly O(1) for general subword acronyms spanning multiple words, unless you limit the prefixes or length of words considered. Any full solution is going to hit the combinatorial wall."



#text = text.lower()
#text2 = text2.lower()

# Step 2: Build all length-2 acrostics
acro_map = defaultdict(list)
acro_map2 = defaultdict(list)
def acro_map_simple(words, n=5):
    acro_map = defaultdict(list)

    for i in range(len(words) - n + 1):
        # concatenate first letters
        acronym = ''.join(words[i+j][0] for j in range(n))
        # concatenate full words
        acrostic = ' '.join(words[i+j] for j in range(n))
        acro_map[acronym].append(acrostic)
    print("same speed")
    return acro_map
    
def compositions(n, k):
    """All ways to split n into k positive integers."""
    if k == 1:
        yield (n,)
    else:
        for i in range(1, n - k + 2):
            for tail in compositions(n - i, k - 1):
                yield (i,) + tail


def acro_map_complex(words, n=5):
    acro_map = defaultdict(list)
    max_words = n

    for i in range(len(words)):
        for span in range(1, min(max_words+1, len(words)-i+1)):
            # generate subword acronym
            combined = ''.join(words[i+j][0] for j in range(span))  # first letters
            if len(combined) >= n:
                symbol = combined[:n]
                acro_map[symbol].append(' '.join(words[i:i+span]))

    return acro_map
    
def acro_map_counts(words, n=5):
    acro_map = defaultdict(list)
    for i in range(len(words) - 1):
        for j in range(2, min(n+1, len(words) - i + 1)):
            sub_words = words[i:i+j]

            for lengths in compositions(n, j):
                if all(len(sub_words[k]) >= lengths[k] for k in range(j)):
                    symbol = ''.join(sub_words[k][:lengths[k]] for k in range(j))
                    acro_counts[symbol] += 1

    print("count only, memory efficient")
    return acro_counts

n = 2
mode = "simple"
#mode = "complex"

text = clean_text(text)
text2 = clean_text(text2)
#'''
lengths_normalized = True
def lengths_normalized(text, text2):
    if len(text) > len(text2):
        del text[len(text2):]
    elif len(text) < len(text2):
        del text2[len(text):]
    
    return text, text2

if lengths_normalized:
    text, text2 = lengths_normalized(text, text2)
#'''
if mode == "simple":
    acro_map = acro_map_simple(text, n)
    acro_map2 = acro_map_simple(text2, n)
elif mode == "complex":
    acro_map = acro_map_complex(text, n)
    acro_map2 = acro_map_complex(text2, n)
else:
    print("'mode' not valid")
    sys.exit()

# Step 3: Count acronyms for each acrostic
acro_counts = {}
acro_counts2 = {}

more_or_less = "more"
than_ = 0
_than_ = more_or_less+"_than_"+str(than_)
if not acro_map:
    print("No acrostics found — check corpus separation.")
    exit()
first_key, first_value = next(iter(acro_map.items()))

more_or_less_count = "more"
min_percent = 0.1
DOMINANCE = 5 / 1
#exclude = ["T", "S"]
exclude = []
exclude_str = "excluding_"+"-".join(exclude) + str(more_or_less_count) + "_min_acronym_frequency_percent_" + str(min_percent)

# Count total acronyms per acrostic
acrostic_totals = {acro: len(acros) for acro, acros in acro_map.items()}
acrostic_totals2 = {acro: len(acros) for acro, acros in acro_map2.items()}

# Find the maximum frequency
max_acro = max(acrostic_totals, key=acrostic_totals.get)
max_acro2 = max(acrostic_totals2, key=acrostic_totals2.get)
max_freq = acrostic_totals[max_acro]
max_freq2 = acrostic_totals2[max_acro2]

if not isinstance(first_value, int):
    for acronym, acrostics in acro_map.items():
        if any(letter in acronym for letter in exclude):
            continue
        counts = Counter(acrostics)
        # Keep all acronyms (or filter frequency >0 or >1 )
        #filtered = {k:v for k,v in counts.items() if v > 0}
        if more_or_less == "more":
            filtered = {k:v for k,v in counts.items() if v > than_}
        elif more_or_less == "less":
            filtered = {k:v for k,v in counts.items() if v < than_}
        else:
            _than_ = "not_filtered"
        # Keep as list of tuples sorted by frequency descending
        total = sum(filtered.values())
        
        # DOMINANCE
        other_counts = Counter(acro_map2.get(acronym, []))
        filtered = {
            k: v
            for k, v in filtered.items()
            if (
                # same acrostic, other corpus
                (o := other_counts.get(k, 0)) == 0
                or max(v, o) / min(v, o) >= DOMINANCE
            )
        }
        
        threshold = max_freq * min_percent
        if more_or_less_count == "more":
            if total > threshold:
                acro_counts[acronym] = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        elif more_or_less_count == "less":
            if total > threshold:
                acro_counts[acronym] = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    print("filtered")
else:
    for acro, count in sorted(acro_map.items(), key=lambda x: x[1], reverse=True):
        acro_counts[acronym] = [("<omitted for memory efficiency>", 1)] * count

first_key, first_value = next(iter(acro_map2.items()))
if not isinstance(first_value, int):
    for acronym, acrostics in acro_map2.items():
        if any(letter in acronym for letter in exclude):
            continue
        counts = Counter(acrostics)
        # Keep all acronyms (or filter frequency >0 or >1)
        #filtered = {k:v for k,v in counts.items() if v > 0}
        if more_or_less == "more":
            filtered = {k:v for k,v in counts.items() if v > than_}
        elif more_or_less == "less":
            filtered = {k:v for k,v in counts.items() if v < than_}
        # Keep as list of tuples sorted by frequency descending
        total = sum(filtered.values())
        
        # DOMINANCE
        other_counts = Counter(acro_map.get(acronym, []))
        filtered = {
            k: v
            for k, v in filtered.items()
            if (
                # same acrostic, other corpus
                (o := other_counts.get(k, 0)) == 0
                or max(v, o) / min(v, o) >= DOMINANCE
            )
        }
        
        threshold = max_freq * min_percent
        if more_or_less_count == "more":
            if total > threshold:
                acro_counts2[acronym] = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        elif more_or_less_count == "less":
            if total > threshold:
                acro_counts2[acronym] = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    print("filtered")
else: 
    for acro, count in sorted(acro_map2.items(), key=lambda x: x[1], reverse=True):
        acro_counts2[acronym] = [("<omitted for memory efficiency>", 1)] * count


# Step 4: Sort acrostics by total frequency descending
sorted_acros = sorted(
    acro_counts.items(),
    key=lambda x: sum(freq for _, freq in x[1]),
)
sorted_acros2 = sorted(
    acro_counts2.items(),
    key=lambda x: sum(freq for _, freq in x[1]),
)

# Step 5: Fast processing and divergence analysis

# Convert sorted lists back into Counters for quick access
sorted_counts = {acro: Counter(dict(acrs)) for acro, acrs in sorted_acros}
sorted_counts2 = {acro: Counter(dict(acrs)) for acro, acrs in sorted_acros2}
combined_counts = defaultdict(Counter)

# Merge dictionaries once
for acro, counter in sorted_counts.items():
    combined_counts[acro].update(counter)
for acro, counter in sorted_counts2.items():
    combined_counts[acro].update(counter)
    
combined_counts = {
    acro: counter
    for acro, counter in combined_counts.items()
    if sum(counter.values()) > 0 and acro.isalpha() and acro[::-1] in combined_counts
}

# Precompute totals
acro_freq_total = sum(sum(counter.values()) for counter in combined_counts.values())
acro_freq_total_1 = sum(sum(counter.values()) for counter in sorted_counts.values())
acro_freq_total_2 = sum(sum(counter.values()) for counter in sorted_counts2.values())

p_aggregate_total = 0
p_aggregate_total_r_comp = 0
small_contributor_freq = 0
large_contributor_freq = 0
small_contributor_count = 0
large_contributor_count = 0

p_aggregate_total_weighted_count = 0
p_aggregate_total_weighted_freq = 0

p_aggregate_total_weighted_count_r_comp = 0
p_aggregate_total_weighted_freq_r_comp = 0

aggritate_filter = 1 # - 0.25
acrostic_list_filter = 1

output_file = f"{file_name[:-4]}_{file_name2[:-4]}_len_total_{str(len(text) + len(text2))}_len_acronym_{n}_min_percent_{min_percent}_{mode}_{aggritate_filter}_{_than_}_ALF-{acrostic_list_filter}_{exclude_str}_lengths_normalized_{bool(lengths_normalized)}_DOMINANCE_{str(DOMINANCE)}.txt"
MAX_FILENAME = 240  # safe on Windows

import hashlib
def safe_filename(name: str, ext: str = ".txt") -> str:
    # strip extension if already present
    if name.endswith(ext):
        name = name[:-len(ext)]

    full_len = len(name) + len(ext)
    if full_len <= MAX_FILENAME:
        return name + ext

    # hash the full original name
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]

    return f"{h}{ext}"
output_file_old = output_file
output_file = safe_filename(output_file)


acro_amount_total = len(combined_counts)
idealist_list = []
physicalist_list = []

# collect all acrostics across all acronyms
all_acrostics = set()
for counter_combined in combined_counts.values():
    all_acrostics.update(counter_combined.keys())

acro_count_total = len(all_acrostics)

acro_count_average = acro_count_total / acro_amount_total
acro_freq_average = acro_freq_total / acro_amount_total

d = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10, "K": 11, "L": 12, "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20, "U": 21, "V": 22, "W": 23, "X": 24, "Y": 25, "Z": 26}


ATT_count = 0
with open(output_file, "w", encoding="utf-8") as f:
    print(output_file)
    f.write(output_file)
    if output_file != output_file_old:
        print(output_file_old)
        f.write(output_file_old)
        print()
        f.write("\n")
    for acro, counter_combined in sorted(combined_counts.items(), key=lambda x: sum(x[1].values())):
        if not acro.isalpha():
            continue
            
        if not acro[::-1] in combined_counts:
            continue
        counts1 = sorted_counts.get(acro, Counter())
        counts2 = sorted_counts2.get(acro, Counter())
        
        counts1_ = sorted_counts.get(acro[::-1], Counter())
        counts2_ = sorted_counts2.get(acro[::-1], Counter())

        acro_freq_1 = sum(counts1.values())
        acro_freq_2 = sum(counts2.values())
        acro_freq = acro_freq_1 + acro_freq_2
        
        acro_freq_1_ = sum(counts1_.values())
        acro_freq_2_ = sum(counts2_.values())
        acro_freq_ = acro_freq_1_ + acro_freq_2_
        
        acro_count_1 = len(counts1)
        acro_count_2 = len(counts2)
        acro_count = len(counter_combined)
        
        acro_count_1_ = len(counts1_)
        acro_count_2_ = len(counts2_)
        
        for acro_, counter_combined_ in sorted(combined_counts.items(), key=lambda x: sum(x[1].values())):
            if acro_ == acro[::-1]:
                acro_count_ = len(counter_combined_)
                break
        
        if(acro_freq_total == 0):
            print("files empty")
            print("files empty")
            print("files empty")
            break

        p1 = (acro_freq_1 / acro_freq_total_1 * 100) if acro_freq_total_1 else 0
        p2 = (acro_freq_2 / acro_freq_total_2 * 100) if acro_freq_total_2 else 0
        p_count = acro_freq / acro_freq_total * 100
        
        p1_ = (acro_freq_1_ / acro_freq_total_1 * 100) if acro_freq_total_1 else 0
        p2_ = (acro_freq_2_ / acro_freq_total_2 * 100) if acro_freq_total_2 else 0
        p_count_ = acro_freq_ / acro_freq_total * 100
        
        if p1 == 0 and p2 == 0:
            continue
        else:
            p_aggregate = (p1 - p2) / (p1 + p2)# if p1 == 0 or p2 == 0
        
        if p1_ == 0 and p2_ == 0:
            continue
        else:
            p_aggregate_ = (p1_ - p2_) / (p1_ + p2_) # if p1 == 0 or p2 == 0 else 
        
        if 1 - abs(p_aggregate) <= aggritate_filter:
            
            print(f"Sorting for divergences, where 1 - abs(p_aggregate) < {aggritate_filter}")
            print(f"Acrostic: {acro} {acro_freq} / {acro_freq_total} * 100% = {p_count}%")
            print(f"{file_name[:-4]} vs {file_name2[:-4]} Acrostic: {acro} {p1}% - {p2}% = {p1 - p2}%")
            print(f"Aggregate = {p_aggregate}")
            print(f"Aggregate_ = {p_aggregate_}")
            
            f.write(f"Sorting for divergences, where p_aggregate < 0.97\n")
            f.write(f"Acrostic: {acro} {acro_freq} / {acro_freq_total} * 100% = {p_count}%\n")
            f.write(f"{file_name[:-4]} vs {file_name2[:-4]} Acrostic: {acro} {p1}% - {p2}% = {p1 - p2}%\n")
            f.write(f"Aggregate = {p_aggregate}\n")
            f.write(f"Aggregate_ = {p_aggregate_}\n")

            
            if(p_aggregate > 0):
                idealist_list.append(acro)
            elif(p_aggregate < 0):
                physicalist_list.append(acro)
            else:
                print("BROKEN")
                break
            
                
                '''
                if(acro[0] in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]):
                    p_aggregate_total_r_comp += p_aggregate - p_aggregate_
                else:
                    p_aggregate_total_r_comp -= p_aggregate - p_aggregate_
                '''
            
            # Print top 3 contributors
            for i, (acrostic, freq) in enumerate(counter_combined.most_common()):
                if i < acrostic_list_filter:
                    c1 = counts1[acrostic] if acrostic in counts1 else "None"
                    c2 = counts2[acrostic] if acrostic in counts2 else "None"
                    print(f" {acrostic}: {freq} {file_name[:-4]}: {c1} {file_name2[:-4]}: {c2}")
                    f.write(f"  {acrostic}: {freq}  {file_name[:-4]}: {c1}  {file_name2[:-4]}: {c2}\n")
                    
                    large_contributor_count += 1
                    large_contributor_freq += freq
                else:
                    small_contributor_count += 1
                    small_contributor_freq += freq

                # Weighted sums
            p_aggregate_total += abs(p_aggregate)
            aggrigate_weighted_freq = p_aggregate * acro_freq / acro_freq_average
            aggrigate_weighted_count = p_aggregate * acro_count / acro_count_average
            p_aggregate_total_weighted_freq += abs(aggrigate_weighted_freq)
            p_aggregate_total_weighted_count += abs(aggrigate_weighted_count)

            print(f"Aggregate weighted freq = {aggrigate_weighted_freq}")
            print(f"Aggregate weighted count = {aggrigate_weighted_count}")
            f.write(f"Aggregate weighted freq = {aggrigate_weighted_freq}\n")
            f.write(f"Aggregate weighted count = {aggrigate_weighted_count}\n")
            
            aggrigate_weighted_freq_ = p_aggregate_ * acro_freq_ / acro_freq_average
            aggrigate_weighted_count_ = p_aggregate_ * acro_count_ / acro_count_average
            if "".join(sorted(acro)) == acro and acro != acro[::-1]:
                ATT_count += 1
                denom = abs(p_aggregate) + abs(p_aggregate_)
                p_aggregate_total_r_comp += (p_aggregate - p_aggregate_) / denom if denom != 0 else 0 #* abs(d[acro[1]] - d[acro[0]])/13
                denom = abs(aggrigate_weighted_freq) + abs(aggrigate_weighted_freq_)
                p_aggregate_total_weighted_freq_r_comp += (aggrigate_weighted_freq - aggrigate_weighted_freq_) / denom if denom != 0 else 0 #* abs(d[acro[1]] - d[acro[0]])/13
                denom = abs(aggrigate_weighted_count) + abs(aggrigate_weighted_count_)
                p_aggregate_total_weighted_count_r_comp += (aggrigate_weighted_count - aggrigate_weighted_count_) / denom if denom != 0 else 0 #* abs(d[acro[1]] - d[acro[0]])/13
            elif "".join(sorted(acro)) == acro[::-1] and acro != acro[::-1]:
                ATT_count += 1
                denom = abs(p_aggregate_) + abs(p_aggregate)
                p_aggregate_total_r_comp += (p_aggregate_ - p_aggregate) / denom if denom != 0 else 0 #* abs(d[acro[1]] - d[acro[0]])/13
                denom = abs(aggrigate_weighted_freq_) + abs(aggrigate_weighted_freq)
                p_aggregate_total_weighted_freq_r_comp += (aggrigate_weighted_freq_ - aggrigate_weighted_freq) / denom if denom != 0 else 0 #* abs(d[acro[1]] - d[acro[0]])/13
                denom = abs(aggrigate_weighted_count_) + abs(aggrigate_weighted_count)
                p_aggregate_total_weighted_count_r_comp += (aggrigate_weighted_count_ - aggrigate_weighted_count) / denom if denom != 0 else 0 #* abs(d[acro[1]] - d[acro[0]])/13

            
            print(f"Aggregate_ weighted freq = {aggrigate_weighted_freq_}")
            print(f"Aggregate_ weighted count = {aggrigate_weighted_count_}")
            f.write(f"Aggregate_ weighted freq = {aggrigate_weighted_freq_}\n")
            f.write(f"Aggregate_ weighted count = {aggrigate_weighted_count_}\n")
            
            
            if p1 > p2:
                print(f"Favored by: {file_name[:-4]}")
                f.write(f"Favored by: {file_name[:-4]}\n\n\n")
            elif p1 < p2:
                print(f"Favored by: {file_name2[:-4]}")
                f.write(f"Favored by: {file_name2[:-4]}\n\n\n")
            elif p1 == p2:
                print(f"Favored by: nobody in particular")
                f.write(f"Favored by: nobody in particular\n\n\n")
            print()

    # Summary
    print()
    # print(f"p_aggregate_total = {p_aggregate_total}")
    if acro_count:
        print(f"p_aggregate_abs_average = {p_aggregate_total / acro_amount_total}")
        print(f"p_aggregate_abs_average_weighted_count = {p_aggregate_total_weighted_count} / {acro_amount_total} = {p_aggregate_total_weighted_count / acro_amount_total}")
        print(f"p_aggregate_abs_average_weighted_freq = {p_aggregate_total_weighted_freq / acro_amount_total}")
        
        f.write(f"p_aggregate_abs_average = {p_aggregate_total / acro_amount_total}\n")
        f.write(f"p_aggregate_abs_average_weighted_count = {p_aggregate_total_weighted_count} / {acro_amount_total} = {p_aggregate_total_weighted_count / acro_amount_total}\n")
        f.write(f"p_aggregate_abs_average_weighted_freq = {p_aggregate_total_weighted_freq / acro_amount_total}\n")
        
        
        print(f"p_aggregate_average_r_comp = {p_aggregate_total_r_comp / ATT_count}")
        print(f"p_aggregate_average_weighted_count_r_comp = {p_aggregate_total_weighted_count_r_comp} / {ATT_count} = {p_aggregate_total_weighted_count_r_comp / ATT_count}")
        print(f"p_aggregate_average_weighted_freq_r_comp = {p_aggregate_total_weighted_freq_r_comp / ATT_count}")
        
        f.write(f"p_aggregate_average_r_comp = {p_aggregate_total_r_comp / ATT_count}\n")
        f.write(f"p_aggregate_average_weighted_count_r_comp = {p_aggregate_total_weighted_count_r_comp} / {ATT_count} = {p_aggregate_total_weighted_count_r_comp / ATT_count}\n")
        f.write(f"p_aggregate_average_weighted_freq_r_comp = {p_aggregate_total_weighted_freq_r_comp / ATT_count}\n")
    else:
        print("Sort too selective")
        f.write("Sort too selelctive")
    
    print()
    print("FIX THE COMPLEX FUNCTION, IT'S saying total freq is 1/6 of simple for n=2")
    print("also it's only doing 2 length acrostics for n=5")
    print("and refine the definition so maybe I include full words and only single word length acrostics")
    print("REIMPLEMENT THE COUNT MATRIX WITHOUT SAVING SPECIFIC ACRONYMS")
    print("WRITE A LOOKUP FUNCTION FOR ANY ACRONYM COUNTS THAT SEEM INTERESTING")
    print("compare # files, not just 2")
    print("create more stats, meta stats about whether there's a tendency for Z->A direction acronyms to be more materialist relative to A->Z")
    print("what about A->Z->A or ZAZ and the like, getting more complex?")
    print()
    f.write("\n")
    print("len(text) "+str(len(text)))
    f.write("len(text) "+str(len(text)))
    print("len(text2) "+str(len(text2)))
    f.write("len(text2) "+str(len(text2)))
    
    i_difference = 0
    i_height = 0
    p_difference = 0
    p_height = 0
    print(f"{file_name}:")
    for item in idealist_list:
        WIDTH_ITEM  = len(item) + 1
        WIDTH_DELTA = 5

        item_str  = f"{str(item):<{WIDTH_ITEM}}"[:WIDTH_ITEM]
        delta_str = f"{d[item[1]] - d[item[0]]:<{WIDTH_DELTA}}"[:WIDTH_DELTA]
        
        print(item_str + delta_str,end='')
        f.write(item_str + delta_str)
        
        i_difference += d[item[1]] - d[item[0]]
        i_height += d[item[1]] + d[item[0]]
    print()
    f.write("\n")
    print(str(i_difference/len(idealist_list)))
    f.write(str(i_difference/len(idealist_list)))
    print(str(i_height/len(idealist_list) / n))
    f.write(str(i_height/len(idealist_list) / n))
    print()
    print()
    print()
    f.write("\n")
    f.write("\n")
    f.write("\n")
    print(f"{file_name2}:")
    for item in physicalist_list:
        WIDTH_ITEM  = len(item) + 1
        WIDTH_DELTA = 5

        item_str  = f"{str(item):<{WIDTH_ITEM}}"[:WIDTH_ITEM]
        delta_str = f"{d[item[1]] - d[item[0]]:<{WIDTH_DELTA}}"[:WIDTH_DELTA]
        
        print(item_str + delta_str,end='')
        f.write(item_str + delta_str)
        
        p_difference += d[item[1]] - d[item[0]]
        p_height += d[item[1]] + d[item[0]]
    print()
    f.write("\n")
    print(str(p_difference/len(physicalist_list)))
    f.write(str(p_difference/len(physicalist_list)))
    print(str(p_height/len(physicalist_list) / n))
    f.write(str(p_height/len(physicalist_list) / n))
    print()
    f.write("\n")
    print(file_name)
    f.write(file_name)
    print(file_name2)
    f.write(file_name2)

    if small_contributor_freq:
        print(f"Small Contributor - Count: {small_contributor_count}  Freq: {small_contributor_freq}  Fraction: {small_contributor_count / small_contributor_freq}")
        f.write(f"Small Contributor - Count: {small_contributor_count}  Freq: {small_contributor_freq}  Fraction: {small_contributor_count / small_contributor_freq}")
    if large_contributor_freq:
        print(f"Large Contributor - Count: {large_contributor_count}  Freq: {large_contributor_freq}  Fraction: {large_contributor_count / large_contributor_freq}")
        f.write(f"Large Contributor - Count: {large_contributor_count}  Freq: {large_contributor_freq}  Fraction: {large_contributor_count / large_contributor_freq}")
    print()
    print(output_file)
    f.write(output_file)
    if output_file != output_file_old:
        print(output_file_old)
        f.write(output_file_old)