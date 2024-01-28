import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':

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

    data = []
    for category in categories:
        for index in results.keys():
            data.append({"label": category, "fscore": results[index]['label_score'][category]})

        df = pd.DataFrame(data)

        # Plot using seaborn
        plt.figure(figsize=(12, 10))  # Adjust size as needed

        sns.boxplot(x="label", y="fscore", data=df)
        plt.rcParams.update({'font.size': 14})  # Increase as needed
        plt.xlabel('Label', fontsize=18)
        plt.ylabel('F-score', fontsize=18)
        plt.xticks(rotation=90)  # This rotates the x-axis labels so they don't overlap
        plt.tight_layout()
        plt.savefig("summary_boxplot.svg", format="svg")  # Save the plot as an SVG file
        plt.show()