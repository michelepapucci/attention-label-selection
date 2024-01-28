import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import classification_report
from scipy.stats import spearmanr


def extract_rank_x_repr(rank, ranked_label):
    return {key: (ranked_label[key][rank]) for key in ranked_label}


if __name__ == "__main__":
    ranks = []
    fscores = []

    folder = "results_on_eos_full_dataset"

    representations = f"{folder}/top_words_100.json"
    with open(representations) as input_file:
        rankings = json.loads(input_file.read())

    total_results = {}

    for index in range(23):
        total_results[index] = {
            "weighted_f1-score": 0,
            "label_score": {}
        }
        index_text = f"_{index}"
        predictions = pd.read_csv(
            f"{folder}/t5_predictions_23/prediction_rank{index_text}.tsv",
            sep="\t")
        repr = extract_rank_x_repr(index, rankings)
        report = classification_report(predictions['y_true'], predictions['y_pred'], output_dict=True)
        ranks.append(index)
        fscores.append(report['weighted avg']['f1-score'])
        for label in rankings:
            try:
                total_results[index]["weighted_f1-score"] = report['weighted avg']["f1-score"]
                total_results[index]["label_score"][label] = report[repr[label]]["f1-score"]
            except Exception as e:
                print(index)
                print(e)
                print(label)
                print(repr[label])
                print(report)

    with open("total_results.json", "w") as output:
        output.write(json.dumps(total_results))

    df = pd.DataFrame({"ranks": ranks, 'score': fscores})
    sns.scatterplot(data=df, x='ranks', y='score')  # changed this line
    plt.title(f"Ranks x f-score")
    plt.savefig("rank_by_score.svg", format="svg")  # Save the plot as an SVG file

    sns.regplot(data=df, x='ranks', y='score')  # changed this line
    plt.title(f"Ranks x f-score")
    X = df['ranks']
    y = df['score']

    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())
    print(spearmanr(df['ranks'], df['score']))
    plt.savefig("rank_by_score_regression_line.svg", format="svg")  # Save the plot as an SVG file
    with open(f"rank_by_score_report.txt", "w") as output_file:
        output_file.write(str(results.summary()))
        output_file.write("\n\nSpearman\n\n")
        output_file.write(str(spearmanr(df['ranks'], df['score'])))
    plt.show()
    maxrank = 0
    maxscore = 0
    minrank = 101
    minscore = 1000
    for rank, score in zip(ranks, fscores):
        if score > maxscore:
            maxscore = score
            maxrank = rank
        if score < minscore:
            minscore = score
            minrank = rank
        print(f"{rank}: {score}\n")
    print(f"Max rank: {maxrank}, score: {maxscore}\n")
    print(f"Min rank: {minrank}, score: {minscore}")
