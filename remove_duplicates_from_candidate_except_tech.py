import json

if __name__ == "__main__":

    # 1. Load the JSON data
    with open('sorted_uncleaned_data.json', 'r') as f:
        data = json.load(f)

    # 2. Iterate through the categories and store the key frequencies and categories they appear in
    key_frequencies = {}
    key_categories = {}
    for category, words in data.items():
        for word in words:
            key_frequencies[word] = key_frequencies.get(word, 0) + 1
            if word not in key_categories:
                key_categories[word] = []
            key_categories[word].append(category)

    # 3. Create two dictionaries: cleaned data and removed keys
    cleaned_data = {}
    removed_keys = {}

    for category, words in data.items():
        cleaned_data[category] = {}
        removed_keys[category] = []

        for word, values in words.items():
            # If word appears only once across all categories or if the word is in "TECHNOLOGY", add it to cleaned data
            if key_frequencies[word] == 1 or (category == "TECHNOLOGY" and key_frequencies[word] > 1):
                cleaned_data[category][word] = values
            elif category != "TECHNOLOGY" and "TECHNOLOGY" in key_categories[word]:
                removed_keys[category].append(word)
            elif key_frequencies[word] > 1:
                removed_keys[category].append(word)

    # 4. Write the dictionaries to two separate JSON files
    with open('cleaned_data.json', 'w') as f:
        json.dump(cleaned_data, f, indent=4)

    with open('removed_keys_data.json', 'w') as f:
        json.dump(removed_keys, f, indent=4)
