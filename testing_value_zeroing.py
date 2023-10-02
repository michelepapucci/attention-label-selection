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
                                        tokenizer.pad_token_id)  # Since we are not generating it's a pad token tensor

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

    og_label_map = {
        'ANIME': 'Anime',
        'BIKES': 'Bicicletta',
        'SPORTS': 'Sport',
        'AUTO-MOTO': 'Automobilismo',
        'NATURE': 'Natura',
        'METAL-DETECTING': 'Metal Detector',
        'MEDICINE-AESTHETICS': 'Medicina',
        'CELEBRITIES': 'Celebrità',
        'SMOKE': 'Fumo',
        'ENTERTAINMENT': 'Intrattenimento',
        'TECHNOLOGY': 'Tecnologia'
    }

    text = "Glaceon  lv28 ps119 Abilità: Mantelneve  Alitogelido (-60, brutto colpo) Ventogelato (-59, riduce Vel) Morso (-62, può far Tent) Gelodenti (-66, può far Tent o Gel) Exp: 30/100 Morso e exeggcute KO ricerco però stavolta erba livello 35"
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # text = " ".join(row['Sentence'].split(" ")[:200]) + "\n" + og_label_map[row['Topic']]
        inputs = tokenizer(text, return_tensors="pt", max_length=256,
                           truncation=True)

        test_label_inputs = tokenizer("Anime", return_tensors="pt", max_length=256,
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
        print(tokenized_tokens)

        _, rollout_scores = compute_sentence_rollout_attention(concat_inputs, model, tokenizer, config)

        # Getting only the last layer, and removing the EOS tokens for Viz.
        rollout_scores = rollout_scores[-1][:-1, :-1]

        print(tokenized_tokens)

        label_tokens = [tokenizer.convert_ids_to_tokens(t) for t in test_label_inputs['input_ids']][0][:-1]
        rollout = pd.DataFrame(rollout_scores, columns=tokenized_tokens[0][:-1], index=tokenized_tokens[0][:-1])

        # getting only the part regarding the label tokens.
        rollout = rollout.tail(test_label_inputs['input_ids'].shape[1] - 1)

        print(rollout)

        sns.heatmap(data=rollout, cmap='Blues', annot=False, cbar=False, xticklabels=True,
                    yticklabels=True)

        plt.show()
        # print(df[test_label_inputs['input_ids']])
        break

"""def preprocess_data(examples):   
    inputs = [text for text in examples["Sentence"] if text != None]

    # Setup the tokenizer for targets
    texts_target = [label_repr[text][0] for text in examples[label_column] if text != None]

    model_inputs = tokenizer(inputs, max_length=max_input_length, padding='max_length',
                             truncation=True)
    model_inputs['labels'] = tokenizer(text_target=texts_target, max_length=max_target_length,
                                       padding='max_length', truncation=True)['input_ids']
    return model_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Calculate accuracy and f-score for each class
    report = classification_report(decoded_preds, decoded_labels, output_dict=True)

    return {
        'accuracy': report['accuracy'],
        'f1_score_macro': report['macro avg']['f1-score'],
        'f1_score_weighted': report['weighted avg']['f1-score'],
    }


model_dir = f"models/fine-tuned-model"

model_args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="epoch",
    eval_steps=50000,
    logging_strategy="epoch",
    logging_steps=1000,
    save_strategy="epoch",
    save_steps=50000,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=epochs,
    predict_with_generate=True,
    fp16=False
)

test_df = pd.read_csv(test_df_path)
test = Dataset.from_pandas(test_df)
train = Dataset.from_pandas(df)

# Tokenize data
tokenized_train = train.map(preprocess_data,
                            batched=True)

tokenized_test = test.map(preprocess_data,
                          batched=True)

train_dataset = tokenized_train.remove_columns(
    [column for column in tokenized_train.features if column not in ['attention_mask', 'labels', 'input_ids']])
test_dataset = tokenized_test.remove_columns(
    [column for column in tokenized_test.features if column not in ['attention_mask', 'labels', 'input_ids']])

train_dataset = train_dataset.shuffle(seed=42)
test_dataset = test_dataset.shuffle(seed=42)
"""
