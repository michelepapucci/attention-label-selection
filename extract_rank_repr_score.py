import argparse
import json
import pandas as pd
from sklearn.metrics import classification_report


def extract_rank_x_repr(rank, ranked_label):
    return {key: (ranked_label[key][rank]) for key in ranked_label}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help="input repr file")
    parser.add_argument('rank', type=int, help='Rank to extract from')

    with open(parser.parse_args().input_file) as input_file:
        rankings = json.loads(input_file.read())

    print(extract_rank_x_repr(parser.parse_args().rank, rankings))
    predictions = pd.read_csv(
        f"results_on_eos_full_dataset/t5_prediction_no_tech_100/prediction_rank_{parser.parse_args().rank}.tsv",
        sep="\t")
    print(classification_report(predictions['y_true'], predictions['y_pred']))
