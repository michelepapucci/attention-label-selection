import json


def average_second_value(values):
    """Compute the average of the second values in a list of lists."""
    return sum(value[1] for value in values) / len(values)


if __name__ == "__main__":
    # Load the JSON data
    with open("cleaned_data.json", "r") as file:
        data = json.load(file)

    # Sort each topic's keys based on the product of the average second value and the frequency
    sorted_data = {}
    for topic, keywords in data.items():
        sorted_keywords = dict(
            sorted(
                keywords.items(),
                key=lambda item: average_second_value(item[1]) * len(item[1]),
                reverse=True
            )
        )
        sorted_data[topic] = sorted_keywords

    # Save the sorted data back to JSON if necessary
    with open("sorted_cleaned_data.json", "w") as file:
        json.dump(sorted_data, file, indent=4)

    print("Data has been sorted and saved to 'sorted_data.json'")
