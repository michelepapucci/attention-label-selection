import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("data/training_filtered.csv")
    df = df[df["Topic"] != "Technology"]
    df.to_csv("data/training_filtered_no_tech.csv", index=False)
    df = pd.read_csv("data/training_filtered.csv")
    df = df[df["Topic"] != "Technology"]
    df.to_csv("data/test_1_filtered_no_tech.csv", index=False)
