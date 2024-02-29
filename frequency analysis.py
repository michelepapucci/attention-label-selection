import json
from scipy.stats import spearmanr

training_set = open("data/training_filtered_no_tech.csv").read()


def calculate_frequency(repr):
    return training_set.count(repr)


if __name__ == "__main__":
    rankings = {}
    with open("results_on_eos_full_dataset/top_words_100.json") as input_file:
        rankings = json.loads(input_file.read())

    fscore_results = json.loads(open("results_on_eos_full_dataset/results_rank_fscore_labels_100.json").read())

    fscore_frequency = {}

    for index in range(100):
        for category in rankings:
            if category not in fscore_frequency:
                fscore_frequency[category] = {"Frequency": [], "Score": []}
            fscore_frequency[category]["Score"].append(fscore_results[str(index)]["label_score"][category])
            fscore_frequency[category]["Frequency"].append(calculate_frequency(rankings[category][index]))

    with open("fscore_frequency.json", "w") as output_file:
        output_file.write(json.dumps(fscore_frequency))

    print("SPEARMAN BETWEEN FREQUENCY AND F-SCORE")

    for category in fscore_frequency:
        print(category)
        print(spearmanr(fscore_frequency[category]["Frequency"], fscore_frequency[category]["Score"]))

    print("\n\nSPEARMAN BETWEEN FREQUENCY AND REPRESENTATION RANK")

    for index, category in enumerate(fscore_frequency):
        print(category)
        print(spearmanr(fscore_frequency[category]["Frequency"], range(len(fscore_frequency[category]["Frequency"]))))
