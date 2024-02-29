import json
import string

import pandas as pd
from cleantext import clean
from math import log10

from scipy.stats import spearmanr

df = pd.read_csv("data/training_filtered_no_tech.csv")


def length_of_documents(category):
    category_df = df[df['Topic'] == category]
    length = 0
    for index, row in category_df.iterrows():
        length += len(row['Sentence'].split(" "))
    return length


length_of_category = {category: length_of_documents(category) for category in df['Topic'].unique()}


def calculate_tf_idf(category, term):
    category_df = df[df['Topic'] == category]
    inverse_category_df = df[df['Topic'] != category]
    category_count = category_df['Sentence'].str.contains(term, case=False).sum()
    inverse_category_count = 0
    for c in inverse_category_df['Topic'].unique():
        inverse_c_df = inverse_category_df[inverse_category_df['Topic'] == c]
        tmp = inverse_c_df['Sentence'].str.contains(term, case=False).sum()
        if tmp > 0:
            inverse_category_count += 1

    tf = category_count / length_of_category[category]
    if inverse_category_count == 0:
        inverse_category_count = 1
    idf = log10(9 / inverse_category_count + 1)

    # print(category, term, tf * idf)
    return tf * idf


if __name__ == "__main__":
    rankings = {}
    with open("results_on_eos_full_dataset/top_words_100.json") as input_file:
        rankings = json.loads(input_file.read())

    fscore_results = json.loads(open("results_on_eos_full_dataset/results_rank_fscore_labels_100.json").read())

    fscore_frequency = {}

    for index in range(100):
        for category in rankings:
            if category not in fscore_frequency:
                fscore_frequency[category] = {"TFIDF": [], "Score": []}
            cleaned_word = clean(rankings[category][index].translate(str.maketrans('', '', string.punctuation)),
                                 no_emoji=True,
                                 no_punct=True,
                                 normalize_whitespace=True)
            fscore_frequency[category]["Score"].append(fscore_results[str(index)]["label_score"][category])
            fscore_frequency[category]["TFIDF"].append(calculate_tf_idf(category, cleaned_word))

    print("SPEARMAN BETWEEN FREQUENCY AND F-SCORE")

    for category in fscore_frequency:
        print(category)
        print(spearmanr(fscore_frequency[category]["TFIDF"], fscore_frequency[category]["Score"]))

    print("\n\nSPEARMAN BETWEEN FREQUENCY AND REPRESENTATION RANK")

    for index, category in enumerate(fscore_frequency):
        print(category)
        print(spearmanr(fscore_frequency[category]["TFIDF"], range(len(fscore_frequency[category]["TFIDF"]))))
