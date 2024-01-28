import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import spearmanr

if __name__ == "__main__":
    categories = ["BIKES",
                  "SPORTS",
                  "ANIME",
                  "AUTO-MOTO",
                  "NATURE",
                  "METAL-DETECTING",
                  "MEDICINE-AESTHETICS",
                  "CELEBRITIES",
                  "SMOKE",
                  "ENTERTAINMENT"]

    with open("results_on_eos_full_dataset/results_rank_fscore_labels_100.json") as input_file:
        results = json.loads(input_file.read())

    for category in categories:
        scores = []
        indices = []
        for index in results.keys():
            indices.append(int(index))
            scores.append(results[index]['label_score'][category])

        print(category)
        df = pd.DataFrame({"ranks": indices, 'score': scores})
        sns.scatterplot(data=df, x='ranks', y='score')  # changed this line
        plt.title(f"{category} Ranks x f-score")

        sns.regplot(data=df, x='ranks', y='score')  # changed this line
        X = df['ranks']
        y = df['score']

        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        report = model.fit()
        print(report.summary())
        print(spearmanr(df['ranks'], df['score']))
        plt.savefig(f"rank_by_fscore_{category}.svg", format="svg")  # Save the plot as an SVG file
        plt.show()
        with open(f"rank_by_score_report_{category}.txt", "w") as output_file:
            output_file.write(str(report.summary()))
            output_file.write("\n\nSpearman\n\n")
            output_file.write(str(spearmanr(df['ranks'], df['score'])))
