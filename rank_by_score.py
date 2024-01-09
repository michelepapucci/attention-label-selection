import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import classification_report
from scipy.stats import spearmanr

if __name__ == "__main__":
    ranks = []
    fscores = []

    for index in range(10):
        index_text = f"_{index}"
        predictions = pd.read_csv(
            f"results_on_eos_full_dataset/t5_predictions/prediction_rank{index_text}.tsv",
            sep="\t")

        report = classification_report(predictions['y_true'], predictions['y_pred'], output_dict=True)
        ranks.append(index)
        fscores.append(report['weighted avg']['f1-score'])
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
    with open(f"plots/rank_by_score_report.txt", "w") as output_file:
        output_file.write(str(results.summary()))
        output_file.write("\n\nSpearman\n\n")
        output_file.write(str(spearmanr(df['ranks'], df['score'])))
    plt.show()
