"""
Collection of evaluation metric functions for evaluating topics and sentiments.
"""

import numpy as np
import pandas as pd
import math

from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm


def npmi(w1_freq: int, w2_freq: int, joint_freq: int, N: int) -> float:
    """Compute the NORMALIZED pointwise mutual information (NPMI) for two words given
        their joint and marginal frequencies.

    Args:
        w1_freq (int): Frequency of word 1.
        w2_freq (int): Frequency of word 2.
        joint_freq (int): The joint frequency of the two words.
        N (int): Total number of word occurrences (population size).

    Returns:
        float: npmi(w1, w2)
    """
    joint_probability: float = joint_freq / N

    # if the values never occur together, they have an npmi of -1
    if joint_probability == 0:
        return -1

    # otherwise compute the npmi
    pmi_res: float = pmi(w1_freq, w2_freq, joint_freq, N)
    return pmi_res / (-np.log2(joint_probability))


def nppmi(w1_freq: int, w2_freq: int, joint_freq: int, N: int) -> float:
    """Compute the NORMALIZED POSITIVE pointwise mutual information (NPPMI) for two words given
        their joint and marginal frequencies.

    Args:
        w1_freq (int): Frequency of word 1.
        w2_freq (int): Frequency of word 2.
        joint_freq (int): The joint frequency of the two words.
        N (int): Total number of word occurrences (population size).

    Returns:
        float: nppmi(w1, w2)
    """
    joint_probability: float = joint_freq / N

    # if the values never occur together, they have an nppmi of 0
    if joint_probability == 0:
        return 0

    # otherwise compute the npmi
    ppmi_res: float = ppmi(w1_freq, w2_freq, joint_freq, N)
    return ppmi_res / (-np.log2(joint_probability))


def pmi(w1_freq: int, w2_freq: int, joint_freq: int, N: int) -> float:
    """Compute the pointwise mutual information (PMI) for two words given their joint and marginal frequencies.

    Args:
        w1_freq (int): Frequency of word 1.
        w2_freq (int): Frequency of word 2.
        joint_freq (int): The joint frequency of the two words.
        N (int): Total number of word occurrences (population size).

    Returns:
        float: pmi(w1, w2)
    """
    joint_probability: float = joint_freq / N
    w1_probability: float = w1_freq / N
    w2_probability: float = w2_freq / N
    return np.log2(joint_probability / (w1_probability * w2_probability))


def ppmi(w1_freq: int, w2_freq: int, joint_freq: int, N: int) -> float:
    """Compute the POSITIVE pointwise mutual information (PPMI) for two words given
        their joint and marginal frequencies.

    Args:
        w1_freq (int): Frequency of word 1.
        w2_freq (int): Frequency of word 2.
        joint_freq (int): The joint frequency of the two words.
        N (int): Total number of word occurrences (population size).

    Returns:
        float: ppmi(w1, w2)
    """
    pmi_res: float = pmi(w1_freq, w2_freq, joint_freq, N)
    return max(0, pmi_res)


def compute_topic_npmi(documents: list[str], topic_words: list[str], use_ppmi: bool = True) -> float:
    """Compute the (average) NPMI of a topic given its keywords.

    Args:
        documents (list[str]): The list of documents.
        topic_words (list[str]): The list of keywords for the topic.
        use_ppmi (bool): Flag whether to use the positive PMI or regular PMI for the N(P)PMI computation.

    Returns:
        float: Average NPMI of the topic words.
    """
    # let's compute the occurrence and co-occurrence matrices first
    vectorizer: CountVectorizer = CountVectorizer(ngram_range=(1, 1))  # unigram model
    X: np.ndarray = vectorizer.fit_transform(documents)  # type: ignore
    feature_names: list[str] = vectorizer.get_feature_names_out().tolist()

    # remove words that are not vectorised
    new_words: list = []
    for word in topic_words:
        if word in feature_names:
            new_words.append(word)
    topic_words = new_words
    n_topic_words: int = len(topic_words)

    # create a hash table to get the index of each word in the co-occurrence matrix
    word_idxs: dict[str, int] = {}
    for word in topic_words:
        idx: int = feature_names.index(word)
        word_idxs[word] = idx

    # compute individual occurrences
    word_occurrences: np.ndarray = np.ravel(X.sum(axis=0))

    # compute co-occurrences
    co_occurrences: np.ndarray = (X.T * X)  # this is co-occurrence matrix in sparse csr format
    co_occurrences.setdiag(0)  # type: ignore # sometimes you want to fill same word cooccurence to 0

    # count the total number of words
    total_words: int = X.sum()

    # iterate over all word combinations
    npmi_sum: float = 0
    for j in range(1, n_topic_words):
        for i in range(0, j):
            # compute npmi
            wj_occurrence: int = word_occurrences[word_idxs[topic_words[j]]]
            wi_occurrence: int = word_occurrences[word_idxs[topic_words[i]]]
            joint_occurrence: int = co_occurrences[word_idxs[topic_words[j]], word_idxs[topic_words[i]]]
            if use_ppmi:
                npmi_val: float = nppmi(w1_freq=wj_occurrence, w2_freq=wi_occurrence, joint_freq=joint_occurrence,
                                        N=total_words)
            else:
                npmi_val: float = npmi(w1_freq=wj_occurrence, w2_freq=wi_occurrence, joint_freq=joint_occurrence,
                                       N=total_words)
            npmi_sum += npmi_val

    if n_topic_words < 2:
        return npmi_sum
    else:
        return npmi_sum / math.comb(n_topic_words, 2)


def compute_avg_topic_npmi(documents: list[str], topics: list[list[str]], use_ppmi: bool = True,
                           verbose: bool = True) -> float:
    """Compute the average NPMI over all topics.

    Args:
        documents (list[str]): The corpus of documents.
        topics (list[list[str]]): List of topics where each topic is a list of its keywords, i.e.
            a list of lists of strings.
        use_ppmi (bool): Flag whether to use the positive PMI or regular PMI for the N(P)PMI computation.
        verbose (bool): Flag whether to show the progress bar.

    Returns:
        float: The average NPMI over all topics.
    """
    topic_npmi_vals: list[float] = []
    for topic in tqdm(topics, disable=not verbose):
        topic_npmi: float = compute_topic_npmi(documents=documents, topic_words=topic, use_ppmi=use_ppmi)
        topic_npmi_vals.append(topic_npmi)
    return np.mean(topic_npmi_vals)  # type: ignore


def compute_topic_diversity(topics: list[list[str]]) -> float:
    """Compute the topic diversity, i.e. the fraction of unique words in the topic words.

    Args:
        topics (list[list[str]]): List of topics where each topic is a list of its keywords.
            That is, a list of lists of strings.

    Returns:
        float: The topic diversity, in [0,1].
    """
    # compute the number of unique words
    topic_sets: list[set[str]] = [set(topic) for topic in topics]
    word_union = set.union(*topic_sets)

    # compute the number of topic words
    topic_sizes: list[int] = [len(topic) for topic in topics]
    total_topic_words: int = sum(topic_sizes)

    return len(word_union) / total_topic_words


def highest_value_frequency(series: pd.Series) -> float:
    """Compute the relative frequency of the most common entry in a column.

    Args:
        series (pd.Series): The input series / column.
        relative (bool, optional): Flag whether to use relative frequency. Defaults to True.

    Returns:
        float: The relative frequency of the most common value (mode)
    """
    return series.value_counts(normalize=True, sort=True).iloc[0]
