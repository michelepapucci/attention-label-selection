import json
import string

import pandas as pd
from cleantext import clean
from transformers import AutoTokenizer

from scipy.stats import spearmanr

df = pd.read_csv("data/training_filtered_no_tech.csv")
tokenizer = AutoTokenizer.from_pretrained("gsarti/it5-base")


def subtoken_length(word):
    tokenized_word = tokenizer(text_target=word, max_length=10, padding='max_length', truncation=True)['input_ids']
    print(word)
    cleaned_tokenized_word = [el for el in tokenized_word if el not in [0, 1]]
    return len(cleaned_tokenized_word)


if __name__ == "__main__":
    rankings = {}
    with open("results_on_eos_full_dataset/top_words_100.json") as input_file:
        rankings = json.loads(input_file.read())

    fscore_results = json.loads(open("results_on_eos_full_dataset/results_rank_fscore_labels_100.json").read())

    fscore_frequency = {}

    for index in range(100):
        for category in rankings:
            if category not in fscore_frequency:
                fscore_frequency[category] = {"Subtoken_length": [], "Score": []}
            cleaned_word = clean(rankings[category][index].translate(str.maketrans('', '', string.punctuation)),
                                 no_emoji=True,
                                 no_punct=True,
                                 normalize_whitespace=True)
            fscore_frequency[category]["Score"].append(fscore_results[str(index)]["label_score"][category])
            fscore_frequency[category]["Subtoken_length"].append(subtoken_length(rankings[category][index]))

    print("SPEARMAN BETWEEN Subtoken_length AND F-SCORE")

    for category in fscore_frequency:
        print(category)
        print(spearmanr(fscore_frequency[category]["Subtoken_length"], fscore_frequency[category]["Score"]))

    print("\n\nSPEARMAN BETWEEN Subtoken_length AND REPRESENTATION RANK")

    for index, category in enumerate(fscore_frequency):
        print(category)
        print(spearmanr(fscore_frequency[category]["Subtoken_length"],
                        range(len(fscore_frequency[category]["Subtoken_length"]))))
