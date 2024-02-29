import json
import string
from cleantext import clean
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd


def open_formario(file_path):
    formario = {}
    with open(file_path) as input_file:
        for line in input_file:
            splitted = line.split("\t")
            formario[splitted[0]] = splitted[1:]
    return formario


if __name__ == "__main__":
    folder = "results_on_eos_full_dataset"
    file = "top_words_100.json"

    top_words = json.loads(open(folder + "/" + file).read())
    formario = open_formario("formario.conllu")

    results = {"total": {"UNKNOW": 0}}

    for topic in top_words:
        print("\n\n" + topic)
        results[topic] = {"UNKNOW": 0}
        for word in top_words[topic]:
            cleaned_word = clean(word.translate(str.maketrans('', '', string.punctuation)), no_emoji=True,
                                 no_punct=True,
                                 normalize_whitespace=True)
            if cleaned_word in formario:
                pos = formario[cleaned_word][0].split("@")[0]
                if pos not in results["total"]:
                    results["total"][pos] = 1
                else:
                    results["total"][pos] += 1

                if pos not in results[topic]:
                    results[topic][pos] = 1
                else:
                    results[topic][pos] += 1
            else:
                results["total"]["UNKNOW"] += 1
                results[topic]["UNKNOW"] += 1

                print(cleaned_word)
    with open("pos_analysis.json", "w") as output_file:
        output_file.write(json.dumps(results))

    # Data for the 'total' key
    total_data = results["total"].copy()

    # Data excluding 'total'
    data = results.copy()
    del data["total"]

    # Gather all unique categories
    all_categories = set(total_data.keys())
    for value in data.values():
        all_categories.update(value.keys())

    # Create a consistent color palette for all categories
    color_palette = sns.color_palette("hls", len(all_categories))
    category_colors = dict(zip(all_categories, color_palette))

    # Convert 'total_data' to DataFrame and assign colors
    total_df = pd.DataFrame(list(total_data.items()), columns=['Category', 'Count'])
    total_df['Color'] = total_df['Category'].map(category_colors)

    # Plot for 'total'
    plt.figure(figsize=(10, 8))
    pie, _ = plt.pie(total_df['Count'], startangle=140, colors=total_df['Color'])
    plt.axis('equal')
    plt.title('Total Key Distribution')
    centre_circle = plt.Circle((0, 0), 0.50, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.legend(pie, total_df['Category'], title="Categories", loc="best")
    plt.savefig('total_donut_attention_pos_eos.svg', format='svg')
    plt.show()

    # Plot for individual data items
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    axes = axes.flatten()

    for ax, (key, value) in zip(axes, data.items()):
        df = pd.DataFrame(list(value.items()), columns=['Category', 'Count'])
        df['Color'] = df['Category'].map(category_colors)
        pie, _ = ax.pie(df['Count'], startangle=140, colors=df['Color'])
        ax.axis('equal')
        ax.set_title(key)
        centre_circle = plt.Circle((0, 0), 0.50, fc='white')
        ax.add_artist(centre_circle)

    # Create a single legend for all charts using the 'category_colors' dictionary
    patches = [mpatches.Patch(color=color, label=label) for label, color in category_colors.items()]
    plt.figlegend(patches, category_colors.keys(), loc='lower center', ncol=5, title="Categories")

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    # Save as SVG
    plt.savefig('category_donuts_attention_pos_eos.svg', format='svg')

    plt.show()
