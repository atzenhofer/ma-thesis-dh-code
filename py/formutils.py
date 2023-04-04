from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import pandas as pd
import numpy as np
import re
import nltk
import scipy.stats, math
import gzip, bz2, lzma



# General


def get_time():
    """Returns the current date and time in a pre-defined format.
    """
    time = datetime.today().strftime("%Y-%m-%d-%H%M")
    return time


def has_same_length(a, b):
    """Checks if two input strings have the same length.
    """
    return len(a) == len(b)


def get_diff_list(a, b):
    """Returns the difference between two input lists.
    """
    return list((set(a)) - (set(b)))



# Pre-process


def get_unique_characters(text):
    """Returns a sorted list of unique characters in a string.
    """
    return sorted(list(set(text)))


def get_concordance(text, char, range=50):
    """Returns a substring of a text including a specific character surrounded by square brackets.
    """
    index = text.find(char)
    if index == -1:
        return None
    start_index = max(index - range, 0)
    end_index = min(index + range, len(text))
    conc = text[start_index:end_index]
    conc = conc.replace(char, f" [ {char} ] ")
    return conc


def map_strings(text, mapping):
    """Replaces substrings in a text with a corresponding replacement based on a mapping.
    """
    for key, value in mapping.items():
        text = re.sub(re.escape(key), value, text)
    return text


def explode_columns(dataframe, columns=None):
    """Explodes one or more pandas columns in a DataFrame so each row contains only one object.
    """
    all_columns = dataframe.keys().to_list()
    if columns is None:
        columns = all_columns
    for column in columns:
        dataframe = dataframe.explode(column)
    return dataframe


def column_to_sorted_set(df_column):
    """Returns a sorted list of unique values in a DataFrame column.
    """
    return list(sorted(set(df_column.to_list())))


def get_diff_rows(df, df_column, less_list):
    """Returns rows of a DataFrame whose values of a specific column are contained in a specific list.
    """
    return df[df_column.isin(less_list)]


def filter_df_by_strings(dataframe, column, filter_strings, mode="substrate", regex=False):
    """Filters a DataFrame based on whether values in a specific column (do not) contain specific strings.
    """ 
    if regex:
        filter_regex = "|".join(filter_strings)
    else:
        filter_regex = "|".join(map(re.escape, filter_strings))
    if mode == "substrate":
        return dataframe[~dataframe[column].astype(str).str.contains(filter_regex)]
    elif mode == "filter":
        return dataframe[dataframe[column].astype(str).str.contains(filter_regex)]
    else:
        raise ValueError("Invalid mode.")


def filter_df_by_regex(dataframe, column, regex, mode="substrate"):  # change to compile
    """Filters DataFrame based on whether values in a specific column (do not) match a RegEx.
    """
    if mode == "substrate":
        return dataframe[~dataframe[column].astype(str).str.contains(regex)]
    elif mode == "filter":
        return dataframe[dataframe[column].astype(str).str.contains(regex)]
    else:
        raise ValueError("Invalid mode.")


def delete_substrings(df, column, chars_to_remove):
    """Removes specified characters from each string in a column of a DataFrame.
    """
    df[column] = df[column].apply(lambda x: "".join(["" if char in chars_to_remove else char for char in x]))
    return df


def delete_strings_by_regex(df, column, regex_list, replace=""):
    """Replaces strings in a column of a DataFrame that match a RegEx  with some replacement string.
    """
    for rgx in regex_list:
        df.loc[df[column].str.contains(rgx, regex=True), column] = df[column].str.replace(rgx, replace, regex=True)
    return df


def count_characters(text, mode="lower"):
    """Returns the frequency of each character in a string as a fraction of the total number of characters. 
    Either keeps or converts all (to) lowercase.
    """
    if mode == "lower":
        char_counts = Counter(text.lower())
    elif mode == "keep":
        char_counts = Counter(text)
    else:
        raise ValueError("Invalid mode.")
    result = {}

    for char, count in char_counts.items():
        result[char] = count / len(text)
    return result


def filter_df_by_iqr(dataframe, column, iqr_multiplier=1.5):
    """Filters a DataFrame based on whether values in a column are within an IQR.
    """
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    return dataframe[
        (dataframe[column] >= q1 - iqr_multiplier * iqr) & (dataframe[column] <= q3 + iqr_multiplier * iqr)
    ]



# Measures


def get_compression_ratio(text, algorithm):
    """Returns compression rate of a text using one of three compression algorithms:
    GZIP: DEFLATE; LZ + Huffman
    BZ2: Burrows-Wheeler
    LZMA: Lempel-Ziv-Markov chain Algorithm
    """
    if algorithm == "gzip": 
        compressed = gzip.compress(text.encode())
    elif algorithm == "bz2": 
        compressed = bz2.compress(text.encode())
    elif algorithm == "lzma": 
        compressed = lzma.compress(text.encode())
    else:
        raise ValueError("Invalid algorithm.")
    return len(compressed) / len(text)



# Formulas


def get_top_ngrams(text, n, most_common, count=False):
    """Returns k-most_common n-grams form a text.
    Either as set of n-grams or as list of tuples of n-grams and their counts.
    """
    tokens = nltk.word_tokenize(text)
    ngrams_list = list(nltk.ngrams(tokens, n))
    counter = Counter(ngrams_list)
    if count == False:
        return set([item[0] for item in counter.most_common(most_common)])
    else:
        return [(item[0], item[1]) for item in counter.most_common(most_common)]


def get_coverage_score(text, fixed_ngrams):
    """Returns the coverage score of a text based on the number of fixed n-grams, divided by total n-gram number.
    """
    tokens = nltk.word_tokenize(text)
    n = len(next(iter(fixed_ngrams)))
    text_ngrams = list(nltk.ngrams(tokens, n))
    text_ngrams_counter = Counter(text_ngrams)
    matched_ngrams = sum(text_ngrams_counter[ngram] for ngram in fixed_ngrams)
    total_ngrams = sum(text_ngrams_counter.values())
    if total_ngrams > 0:
        return matched_ngrams / total_ngrams
    else:
        return 0


def get_word_probabilities(tokens):
    """Returns a dictionary of probabilities of each word in a list of tokens, divided by total n-gram number.
    """
    counts = Counter(tokens)
    total_count = len(tokens)
    word_probabilities = {}
    for word, count in counts.items():
        word_probabilities[word] = count / total_count
    return word_probabilities


def get_ngram_probabilities(tokens, n):
    """Returns a dictionary of probabilities of each n-gram in a list of tokens, 
    divided by total number of n-gram options.
    """
    ngrams_list = list(nltk.ngrams(tokens, n))
    counts = Counter(ngrams_list)
    total_count = len(tokens) - n + 1
    ngram_probabilities = {}
    for ngram, count in counts.items():
        ngram_probabilities[ngram] = count / total_count
    return ngram_probabilities


def get_ngram_by_score(text, n, min_freq=1, max_freq=float("inf")):
    """Returns sorted list of n-grams in a text, sorted by score.
    Score is calculated based on joint probabilites of n-gram and individual probabilities of its words.
    """
    tokens = nltk.word_tokenize(text)
    word_probs = get_word_probabilities(tokens)
    ngram_probs = get_ngram_probabilities(tokens, n)
    scores = {}
    for ngram, joint_prob in ngram_probs.items():
        if joint_prob * (len(tokens) - n + 1) >= min_freq and (
            max_freq is None or joint_prob * (len(tokens) - n + 1) <= max_freq
        ):
            individual_probs = []
            for word in ngram:
                individual_probs.append(word_probs[word])
            score = joint_prob / (sum(individual_probs) - joint_prob)
            scores[ngram] = score
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    return sorted_scores
    

def get_skip_grams(sequence, n, k):
    """Yields all skip-grams of n-length with k-skips from given sequence.
    """
    for i in range(len(sequence) - n + 1):
        for item in combinations(range(i + 1, min(i + n + k, len(sequence))), n - 1):
            yield (sequence[i],) + tuple(sequence[j] for j in item)


def g2(skipgram, ngrams, word_freq, total_ngrams):
    """Calculates G² association score for a skipgram (n-tuple).
    Based on observed joint frequency of skip-gram and 
    expected joint frequency under assumption of independence between its constituent words.
    """
    observed_joint_freq = ngrams[skipgram]
    if observed_joint_freq == 0:
        return -math.inf
    expected_joint_freq = 1.0
    for word in skipgram:
        expected_joint_freq = expected_joint_freq * (word_freq[word] / total_ngrams)

    observed_freqs = np.array([observed_joint_freq, total_ngrams - observed_joint_freq])
    expected_freqs = np.array([expected_joint_freq, total_ngrams - expected_joint_freq])

    _, p_value = scipy.stats.power_divergence(observed_freqs, expected_freqs, lambda_=0)
    return -math.log10(p_value) if p_value > 0 else -math.inf


def get_skip_grams_by_score(text, n, k, lower_threshold=1, upper_threshold=float("inf")):
    """Generates skip-grams from a text and calculates their G².
    Returns a sorted list of tuples of the skip-gram, the score, and skipped positions.
    """
    tokens = nltk.word_tokenize(text)
    ngrams_counter = Counter(nltk.ngrams(tokens, n))
    word_freq = Counter(tokens)
    total_ngrams = len(ngrams_counter)
    result = []
    processed_skipgrams = set()
    for skipgram_indices in get_skip_grams(range(len(tokens)), n, k):
        skipgram_tokens = tuple(tokens[i] for i in skipgram_indices)
        
        if skipgram_tokens in processed_skipgrams:
            continue
        processed_skipgrams.add(skipgram_tokens)
        score = g2(skipgram_tokens, ngrams_counter, word_freq, total_ngrams)
        
        if lower_threshold <= score <= upper_threshold:
            skipped_positions = [(i, tokens[i]) 
                                    for i in range(skipgram_indices[0] + 1, skipgram_indices[-1])
                                    if i not in skipgram_indices]

            annotated_skipgram = [tokens[i]
                                    if i in skipgram_indices
                                    else f"({tokens[i]})"
                                        for i in range(skipgram_indices[0], skipgram_indices[-1] + 1)]

            result.append((annotated_skipgram, score, skipped_positions))
    result.sort(key=lambda x: -x[1])
    return result


def get_top_ngrams_by_decade(df, n, most_common):
    """Returns a dictionary of top n-gram for each decade based on a DataFrame.
    """
    df["decade"] = df["year"].apply(lambda x: (x // 10) * 10) 
    df["ngrams"] = df["text"].apply(lambda x: get_top_ngrams(x.lower(), n, most_common))
    decades = df["decade"].unique()
    top_ngrams = {}
    for decade in decades:
        ngrams_list = []
        decade_data = df[df["decade"] == decade]
        for _, row in decade_data.iterrows():
            ngrams_list.extend(row["ngrams"])
        counter = Counter(ngrams_list)
        top_ngrams[decade] = counter.most_common(most_common)
    sorted_top_ngrams = dict(sorted(top_ngrams.items()))
    return sorted_top_ngrams


def segment_ngrams(text, n, percentiles):
    """Tokenizes text and extracts n-grams.
    Returns list of lists containing all n-grams of a segment.
    """
    tokens = nltk.word_tokenize(text.lower())
    num_tokens = len(tokens)
    step = num_tokens // percentiles
    segments = []
    for i in range(percentiles):
        start = i * step
        end = start + step
        segment_tokens = tokens[start:end]
        segment_ngrams = list(nltk.ngrams(segment_tokens, n))
        segments.append(segment_ngrams)
    return segments


def get_segmented_ngrams(df, n=10, percentiles=10, most_common=5):
    """Segment text into percentiles and extract n-grams.
    Returns top n-grams per percentile (segment).
    """
    df[f"segment_ngrams_{n}"] = df["text"].apply(segment_ngrams, n=n, percentiles=percentiles)
    segment_ngrams_agg = defaultdict(Counter)
    for _, row in df.iterrows():
        for i, seg_ngrams in enumerate(row[f"segment_ngrams_{n}"]):
            segment_ngrams_agg[i].update(seg_ngrams)
    segmented_ngrams = {i: seg_ngrams.most_common(most_common) for i, seg_ngrams in segment_ngrams_agg.items()}
    return segmented_ngrams



# flexibility


def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets.
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def get_similar_sequences(
        text,
        n,
        most_common,
        similarity_func=jaccard_similarity,
        lower_threshold=0, 
        upper_threshold=1
    ):
    """Extracts top n-grams and finds similar n-grams inside threshold bounds.
    Returns dictionary of top n-gram keys and similar sequences values.
    """
    most_common_ngrams = get_top_ngrams(text, n, most_common)
    ngrams_list = list(nltk.ngrams(nltk.word_tokenize(text), n))
    ngrams = [" ".join(ngram) for ngram in ngrams_list]
    similar_sequences = {}
    
    for common_ngram in most_common_ngrams:
        common_ngram_set = set(common_ngram)
        sequence_scores = {}
        
        for ngram in ngrams:
            ngram_set = set(ngram.split())
            similarity = similarity_func(common_ngram_set, ngram_set)
            if lower_threshold <= similarity < upper_threshold and ngram != " ".join(common_ngram):
                sequence_scores[ngram] = similarity

        sorted_sequences = sorted(sequence_scores.items(), key=lambda x: x[1], reverse=True)
        similar_sequences[" ".join(common_ngram)] = sorted_sequences
    
    return similar_sequences


def predict_tokens(sequence, tokenizer, fill_mask_basic, fill_mask_advanced):
    """Predicts tokens for sequences based on prediction pipeline.
    Returns a DataFrame with rows for tokens of the input sequence, and columns for predictions and prediction scores.
    """
    words = sequence.split()
    mask_token = tokenizer.mask_token
    results = []
    for i, word in enumerate(words):
        masked_sequence = " ".join(words[:i] + [mask_token] + words[i+1:])
        prediction_basic = fill_mask_basic(masked_sequence)[0]
        prediction_advanced = fill_mask_advanced(masked_sequence)[0]
        results.append({
            "Mask": word,
            "Basic Prediction": prediction_basic["token_str"].strip(),
            "Basic Score": prediction_basic["score"],
            "Advanced Prediction": prediction_advanced["token_str"].strip(),
            "Advanced Score": prediction_advanced["score"]
        })
    return pd.DataFrame(results)


def get_recursive_predictions(sequence, fill_mask_basic, fill_mask_advanced):
    """Fills masks recursively based on prediction pipelines.
    Returns predicted tokens and scores for each mask.
    """
    mask_token = "<mask>"
    results = []
    masked_indices = [i for i, word in enumerate(sequence.split()) if word.lower() == mask_token.lower()]
    filled_sequence = sequence   # Use filled_sequence as original
    while mask_token.lower() in filled_sequence.lower(): # Fill masks until full
        predictions_basic = fill_mask_basic(filled_sequence)
        predictions_advanced = fill_mask_advanced(filled_sequence)

        for i, masked_word_idx in enumerate(masked_indices):
            masked_word = f"<mask_{masked_word_idx}>"
            pred_basic = predictions_basic[i][0]
            pred_advanced = predictions_advanced[i][0]

            results.append({
                "Mask": masked_word,
                "Basic Prediction": pred_basic["token_str"].strip(),
                "Basic Score": pred_basic["score"],
                "Advanced Prediction": pred_advanced["token_str"].strip(),
                "Advanced Score": pred_advanced["score"]
            })

            if pred_advanced["token_str"].strip():
                filled_sequence = filled_sequence.replace(mask_token, f"<{pred_advanced['token_str'].strip()}>", 1)
            elif pred_basic["token_str"].strip():
                filled_sequence = filled_sequence.replace(mask_token, f"<{pred_basic['token_str'].strip()}>", 1)
            else:
                filled_sequence = filled_sequence.replace(mask_token, f"<unfilled_mask_{masked_word_idx}>", 1)

    return pd.DataFrame(results), filled_sequence



# format and export


def ngrams_to_dict(result):
    """Return a dictionary of n-grams and their scores.
    """
    d = {}
    for ngram, score in result:
        key = " ".join(ngram)
        d[key] = score
    return d


def scored_ngrams_to_dict(skip_result):
    """Return a dictionary of candidate n-grams and their scores, extracted from a list of n-grams, scores, and skips.
    """
    candidate_grams = []
    for gram, score, skips in skip_result:
        if skips:
            joined_sequence = " ".join(gram)
            candidate_grams.append(joined_sequence)
            return {" ".join(gram): score for gram, score, skips in skip_result if skips}


def dict_to_dataframe(d, columns):
    """Return a DataFrame from a dictionary.
    """
    return pd.DataFrame(list(d.items()), columns=columns)


def modify_cell(cell):
    """Replace tuple in DataFrame cell with string representation.
    """
    if isinstance(cell, tuple) and len(cell) == 2:
        string = " ".join(cell[0])
        number = cell[1]
        return f"{string}, {number}"
    else:
        return cell


def rename_columns(df):
    """Return a DataFrame with renamed columns starting from 1.
    """
    num_cols = len(df.columns)
    new_cols = list(range(1, num_cols + 1))
    df.columns = new_cols
    return df