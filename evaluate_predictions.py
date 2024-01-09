import json
import os

import pandas as pd
from sklearn.metrics import classification_report

if __name__ == "__main__":
    folder = "results_on_eos_full_dataset"
    prediction_folder = folder + "/t5_predictions"
    attention_frequency_score_file = folder + "sorted_cleaned_data.json"
    top_words_file = folder + "top_words.json"

    for i in range(0, 10):
        df = pd.read_csv(os.path.join(prediction_folder + f"/prediction_rank_{i}.tsv"), sep="\t")
        print(classification_report(y_pred=df["y_pred"], y_true=df["y_true"]))

