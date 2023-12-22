import json


def read_json_file(filename):
    """Read a JSON file and return its content as a dictionary."""
    with open(filename, 'r') as f:
        return json.load(f)


def calculate_shared_percentage(keys1, keys2):
    """Calculate the percentage of shared keys."""
    shared_keys = len(set(keys1) & set(keys2))
    total_keys = len(keys1)
    return (shared_keys / total_keys) * 100 if total_keys != 0 else 0


def compare_files(file1, file2):
    """Compare two JSON files and return the shared key percentages."""
    data1 = read_json_file(file1)
    data2 = read_json_file(file2)

    # Calculate local scores
    local_scores = {}
    for category in data1.keys():
        category_keys1 = data1.get(category, {}).keys()
        category_keys2 = data2.get(category, {}).keys()
        local_scores[category] = calculate_shared_percentage(category_keys1, category_keys2)

    # Calculate global score
    all_keys1 = [key for cat in data1 for key in data1[cat]]
    all_keys2 = [key for cat in data2 for key in data2[cat]]
    global_score = calculate_shared_percentage(all_keys1, all_keys2)

    return global_score, local_scores


if __name__ == "__main__":
    files = ['result_label_appended_full_dataset/sorted_cleaned_data',
             'results_on_eos_full_dataset/sorted_cleaned_data',
             'result_prompt_label_appended_full_dataset/sorted_cleaned_data',
             'sorted_cleaned_data']

    for i, file1 in enumerate(files):
        for j, file2 in enumerate(files):
            if i < j:  # To avoid comparing the same files and duplicate comparisons
                global_score, local_scores = compare_files(file1 + '.json', file2 + '.json')
                print(f"Comparing {file1} with {file2}:")
                print(f"Global shared key percentage: {global_score:.2f}%")
                for category, score in local_scores.items():
                    print(f"Shared keys in {category}: {score:.2f}%")
                print("--------")
