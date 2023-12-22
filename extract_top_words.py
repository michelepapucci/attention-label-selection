import json


def extract_top_low_level_words(json_path, save_path):
    # Load data from the provided JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Function to extract the top 10 low-level words for each high-level class
    def extract(data):
        result = {}
        for high_level, low_level_items in data.items():
            # Limiting to the top 10 low-level words
            low_level_words = list(low_level_items.keys())[:10]
            result[high_level] = low_level_words
        return result

    # Extract and save to JSON
    extracted_data = extract(data)
    with open(save_path, 'w') as file:
        json.dump(extracted_data, file)


if __name__ == "__main__":
    folder = "results_on_eos_full_dataset"
    extract_top_low_level_words(folder + "/sorted_cleaned_data.json", folder + "/top_words.json")
