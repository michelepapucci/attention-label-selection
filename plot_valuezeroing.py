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
    # fig, axs = plt.subplots(3, 4, figsize=(70, 70))

    fig, axs = plt.subplots()

    # Increase hspace and wspace for more distance between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    # for layer in layers:
    # a = layer // 4
    # b = layer % 4

    sns.heatmap(ax=axs,
                data=pd.DataFrame(rollout_valuezeroing_scores[0], index=all_tokens[0],
                                  columns=all_tokens[0]), cmap=cmap, annot=False, cbar=False, xticklabels=True,
                yticklabels=True)
    axs.set_title(f"Value-Zeroing Scores")

    # Reduce font size to ensure text fits
    axs.set_xticklabels(axs.get_xticklabels(), fontsize=12)
    axs.set_yticklabels(axs.get_yticklabels(), rotation=360, fontsize=12)
    axs.set_xlabel('')
    axs.set_ylabel('')

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

    text = "Ciao amici e buon anno...e questo che motore era???"

    text = text[:200]
    inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True)

    # prompt = " La frase precedente appartiene alla categoria "
    # Versione dove si appende la label.

    test_label_inputs = tokenizer("La frase precedente appartiene alla categoria automobilismo", return_tensors="pt", max_length=256,
                                  truncation=True)

    """if (inputs['input_ids'].shape[1] == 256 or
            inputs['input_ids'].shape[1] + test_label_inputs['input_ids'].shape[1] >= 256):
        inputs['input_ids'] = inputs['input_ids'][:, :(256 - test_label_inputs['input_ids'].shape[1])]
    else:
        inputs['input_ids'] = inputs['input_ids'][:, :-1]"""

    """concat_inputs = {
        'input_ids': torch.cat((inputs['input_ids'], test_label_inputs['input_ids']), 1),
    }
    concat_inputs['attention_mask'] = torch.ones(concat_inputs['input_ids'].shape)"""

    tokenized_tokens = [tokenizer.convert_ids_to_tokens(t) for t in inputs['input_ids']]
    _, rollout_scores = compute_sentence_rollout_attention(inputs, model, tokenizer, config, plot=True)

    """tokenized_tokens = [tokenizer.convert_ids_to_tokens(t) for t in inputs['input_ids']]
    _, rollout_scores = compute_sentence_rollout_attention(inputs, model, tokenizer, config)"""

    # Getting only the last layer, and removing the EOS tokens for Viz.
    rollout_scores = rollout_scores[-1][:-1, :-1]
    """rollout_scores = rollout_scores[-1][:, :-1]"""
