from sklearn.metrics import classification_report
from transformers import AutoConfig, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from modeling_value_zeroing_t5 import T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
import json


def compute_joint_attention(att_mat, res=True):
    if res:
        residual_att = np.eye(att_mat.shape[1])[None, ...]
        att_mat = att_mat + residual_att
        att_mat = att_mat / att_mat.sum(axis=-1)[..., None]

    joint_attentions = np.zeros(att_mat.shape)
    layers = joint_attentions.shape[0]
    joint_attentions[0] = att_mat[0]
    for i in np.arange(1, layers):
        joint_attentions[i] = att_mat[i].dot(joint_attentions[i - 1])

    return joint_attentions


def compute_sentence_rollout_attention(inputs, model, tokenizer, config, plot=False, layers=None):
    decoder_input_ids = torch.full_like(inputs['input_ids'],
                                        tokenizer.pad_token_id,
                                        device=device)  # Since we are not generating it's a pad token tensor

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        decoder_input_ids=decoder_input_ids,
                        output_hidden_states=True, output_attentions=False)

    org_hidden_states = torch.stack(outputs.encoder_hidden_states).squeeze(1)
    input_shape = inputs['input_ids'].size()
    batch_size, seq_length = input_shape

    score_matrix = np.zeros((config.num_hidden_layers, seq_length, seq_length))
    if layers is None:
        layers = range(len(model.encoder.block))

    with tqdm(total=len(layers) * seq_length) as pbar:
        for layer_index, layer_module in enumerate(model.encoder.block):
            if layer_index in layers:
                for t in range(seq_length):
                    pbar.update(1)
                    extended_blanking_attention_mask: torch.Tensor = model.get_extended_attention_mask(
                        inputs['attention_mask'], input_shape)
                    with torch.no_grad():
                        layer_outputs = layer_module(org_hidden_states[layer_index].unsqueeze(0),
                                                     # previous layer's original output
                                                     attention_mask=extended_blanking_attention_mask,
                                                     output_attentions=False,
                                                     zero_value_index=t,
                                                     )
                    hidden_states = layer_outputs[0].squeeze().detach().cpu().numpy()
                    # compute similarity between original and new outputs
                    # cosine
                    x = hidden_states
                    y = org_hidden_states[layer_index + 1].detach().cpu().numpy()

                    distances = cosine_distances(x, y).diagonal()
                    score_matrix[layer_index, :, t] = distances

    # Workaround for when we have a sum = 0, dunno why it happens tho.
    sum_values = np.sum(score_matrix, axis=-1, keepdims=True)
    mask = (sum_values != 0)
    valuezeroing_scores = np.where(mask, score_matrix / sum_values, 0)  # or some other default value
    rollout_valuezeroing_scores = compute_joint_attention(valuezeroing_scores, res=False)

    if plot:
        visualize_attention_map(inputs, rollout_valuezeroing_scores, tokenizer, layers)

    return valuezeroing_scores, rollout_valuezeroing_scores


def visualize_attention_map(inputs, rollout_valuezeroing_scores, tokenizer, layers=range(12)):
    cmap = "Blues"
    all_tokens = [tokenizer.convert_ids_to_tokens(t) for t in inputs['input_ids']]

    # Increase figure size for more space
    fig, axs = plt.subplots(3, 4, figsize=(70, 70))

    # Increase hspace and wspace for more distance between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    for layer in layers:
        a = layer // 4
        b = layer % 4

        sns.heatmap(ax=axs[a, b],
                    data=pd.DataFrame(rollout_valuezeroing_scores[layer][:-1, :-1], index=all_tokens[0][:-1],
                                      columns=all_tokens[0][:-1]), cmap=cmap, annot=False, cbar=False, xticklabels=True,
                    yticklabels=True)
        axs[a, b].set_title(f"Layer: {layer + 1}")

        # Reduce font size to ensure text fits
        axs[a, b].set_xticklabels(axs[a, b].get_xticklabels(), fontsize=6)
        axs[a, b].set_yticklabels(axs[a, b].get_yticklabels(), rotation=360, fontsize=6)
        axs[a, b].set_xlabel('')
        axs[a, b].set_ylabel('')

    # Ensure the layout is tight to further avoid overlaps
    # fig.tight_layout()
    print(all_tokens)

    plt.show()


if __name__ == "__main__":

    model_name = "gsarti/it5-base"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(model_name)
    config.output_hidden_states = True
    model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)

    df = pd.read_csv("data/training_filtered.csv")

    """og_label_map = {
        'BIKES': 'Bicicletta',
        'SPORTS': 'Sport',
        'TECHNOLOGY': 'Tecnologia',
        'ANIME': 'Anime',
        'AUTO-MOTO': 'Automobilismo',
        'NATURE': 'Natura',
        'METAL-DETECTING': 'Metal Detector',
        'MEDICINE-AESTHETICS': 'Medicina',
        'CELEBRITIES': 'Celebrità',
        'SMOKE': 'Fumo',
        'ENTERTAINMENT': 'Intrattenimento',
    }"""
    og_label_map = {
        'SPORTS': 'Celebrità',
        'ANIME': 'Tecnologia',
        'ENTERTAINMENT': 'Sport',
    }

    label_representations_candidates = {}

    for key in og_label_map:

        by_topic_df = df[df['Topic'] == key]

        counter = 0

        for index, row in tqdm(by_topic_df.iterrows(), total=int(by_topic_df.shape[0])):
            if row['Topic'] not in label_representations_candidates:
                label_representations_candidates[row['Topic']] = {}

            text = " ".join(row['Sentence'].split(" ")[:200])
            inputs = tokenizer(text, return_tensors="pt", max_length=256,
                               truncation=True)

            # prompt = " La frase precedente appartiene alla categoria "
            # Versione dove si appende la label.
            test_label_inputs = tokenizer(og_label_map[row['Topic']], return_tensors="pt", max_length=256,
                                          truncation=True)

            if (inputs['input_ids'].shape[1] == 256 or
                    inputs['input_ids'].shape[1] + test_label_inputs['input_ids'].shape[1] >= 256):
                inputs['input_ids'] = inputs['input_ids'][:, :(256 - test_label_inputs['input_ids'].shape[1])]
            else:
                inputs['input_ids'] = inputs['input_ids'][:, :-1]

            concat_inputs = {
                'input_ids': torch.cat((inputs['input_ids'], test_label_inputs['input_ids']), 1),
            }
            concat_inputs['attention_mask'] = torch.ones(concat_inputs['input_ids'].shape)

            tokenized_tokens = [tokenizer.convert_ids_to_tokens(t) for t in concat_inputs['input_ids']]
            _, rollout_scores = compute_sentence_rollout_attention(concat_inputs, model, tokenizer, config)

            """tokenized_tokens = [tokenizer.convert_ids_to_tokens(t) for t in inputs['input_ids']]
            _, rollout_scores = compute_sentence_rollout_attention(inputs, model, tokenizer, config)"""

            # Getting only the last layer, and removing the EOS tokens for Viz.
            rollout_scores = rollout_scores[-1][:-1, :-1]
            """rollout_scores = rollout_scores[-1][:, :-1]"""

            print(tokenized_tokens)

            label_tokens = [tokenizer.convert_ids_to_tokens(t) for t in test_label_inputs['input_ids']][0][:-1]

            rollout = pd.DataFrame(rollout_scores,
                                   columns=[tok + "•" + str(index) for index, tok in
                                            enumerate(tokenized_tokens[0][:-1])],
                                   index=tokenized_tokens[0][:-1])

            """rollout = pd.DataFrame(rollout_scores,
                                   columns=[tok + "•" + str(index) for index, tok in
                                            enumerate(tokenized_tokens[0][:-1])],
                                   index=tokenized_tokens[0])"""


            # getting only the part regarding the label tokens.
            rollout = rollout.tail(test_label_inputs['input_ids'].shape[1] - 1)
            """rollout = rollout.tail(1)"""

            # sns.heatmap(data=rollout, cmap='Blues', annot=False, cbar=False, xticklabels=True, yticklabels=True)
            # plt.show()

            max_col_per_row = rollout.idxmax(axis=1)
            max_value_per_row = rollout.max(axis=1)

            result_df = pd.DataFrame({
                'column': max_col_per_row,
                'value': max_value_per_row
            })

            print(result_df)

            attention_peak = result_df['value'].max()


            def merge_columns_and_values(start_col_idx, col_names):
                merged_cols = []
                merged_values = []

                # Traverse forward
                for idx in range(start_col_idx, len(col_names)):
                    col_name = col_names[idx]
                    if col_name.startswith('▁') and idx != start_col_idx:
                        break
                    merged_cols.append(col_name)
                    merged_values.append(rollout[col_name])

                if col_names[start_col_idx].startswith('▁'):
                    return merged_cols, pd.concat(merged_values, axis=1).sum(axis=1)
                # Traverse backward
                for idx in range(start_col_idx - 1, -1, -1):
                    col_name = col_names[idx]
                    if col_name.startswith('▁'):
                        merged_cols.insert(0, col_name)  # prepend column names
                        merged_values.insert(0, rollout[col_name])  # prepend values
                        break
                    merged_cols.insert(0, col_name)  # prepend column names
                    merged_values.insert(0, rollout[col_name])  # prepend values

                # Merged value is a matrix label subtoken x candidate subtoken
                # with the sum(axis=1) we have a vector of label subtoken value zeroing score of
                # aggregated importance of the whole candidate
                # for each subtoken of the label. to avoid giving more score to the labels with the most subtoken, we obtain
                # one scalar by averaging it.

                return merged_cols, pd.concat(merged_values, axis=1).sum(axis=1)


            candidates = []
            # Check each column in result_df
            for label_index, result_df_row in result_df.iterrows():
                col_names = rollout.columns.tolist()

                # Ensure that the column name actually exists in `col_names` to avoid a ValueError
                if result_df_row['column'] in col_names:
                    start_col_idx = col_names.index(result_df_row['column'])

                    # Get merged column names and values
                    merged_cols, merged_values = merge_columns_and_values(start_col_idx, col_names)

                    selected_word = ""
                    for subtoken in merged_cols:
                        subtoken = re.sub(r"•\d+", "", subtoken)
                        subtoken = re.sub(r"▁+", "", subtoken)
                        selected_word += subtoken

                    print(f"For Label subtoken '{label_index}' the selected candidate is '{selected_word}', "
                          f"with zero value of: {sum(merged_values.tolist()) / len(merged_values.tolist())}")

                    # The candidates are (word, average word(merged subtoken) attention for each label subtoken,
                    # maximum attention value found in the word subtoken)
                    candidates.append((selected_word,
                                       sum(merged_values.tolist()) / len(merged_values.tolist()),
                                       attention_peak))

                else:
                    print(f"Column {result_df_row['column']} not found in rollout columns!")

            # ... Remaining code ...

            print(candidates)

            elected = max(candidates, key=lambda x: x[1])

            print(f"The elected word for this sentence is: {elected}")

            if elected[0] not in label_representations_candidates[row['Topic']]:
                label_representations_candidates[row['Topic']][elected[0]] = []
            label_representations_candidates[row['Topic']][elected[0]].append((elected[1], elected[2]))

            counter += 1

            """if counter >= by_topic_df.shape[0] / 10:
                break"""

    with open("candidates.json", "w") as file_output:
        file_output.write(json.dumps(label_representations_candidates))
