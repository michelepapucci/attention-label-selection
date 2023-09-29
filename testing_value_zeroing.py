from sklearn.metrics import classification_report
from transformers import AutoConfig, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from value_zeroing_t5 import T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


model_name = "gsarti/it5-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# df = pd.read_csv(dataset_path)

config = AutoConfig.from_pretrained(model_name)
config.output_hidden_states = True
model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
print(model)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.to(device)

text = "Ieri la Juventus ha vinto contro il napoli in casa 2-0. È la fine per la squadra partenopea?"
inputs = tokenizer(text, return_tensors="pt")
decoder_input_ids = torch.full_like(inputs['input_ids'], tokenizer.pad_token_id)

inputs = {k: v.to(device) for k, v in inputs.items()}
print(inputs)
with torch.no_grad():
    outputs = model(inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    decoder_input_ids=decoder_input_ids,  # TODO: gtp-4 does this make sense?
                    output_hidden_states=True, output_attentions=False)

org_hidden_states = torch.stack(outputs.encoder_hidden_states).squeeze(1)
input_shape = inputs['input_ids'].size()
batch_size, seq_length = input_shape

score_matrix = np.zeros((config.num_hidden_layers, seq_length, seq_length))
for l, layer_module in enumerate(model.encoder.block):
    for t in range(seq_length):
        extended_blanking_attention_mask: torch.Tensor = model.get_extended_attention_mask(
            inputs['attention_mask'], input_shape)
        with torch.no_grad():
            layer_outputs = layer_module(org_hidden_states[l].unsqueeze(0),  # previous layer's original output
                                         attention_mask=extended_blanking_attention_mask,
                                         output_attentions=False,
                                         zero_value_index=t,
                                         )
        hidden_states = layer_outputs[0].squeeze().detach().cpu().numpy()
        # compute similarity between original and new outputs
        # cosine
        x = hidden_states
        y = org_hidden_states[l + 1].detach().cpu().numpy()

        distances = cosine_distances(x, y).diagonal()
        score_matrix[l, :, t] = distances

valuezeroing_scores = score_matrix / np.sum(score_matrix, axis=-1, keepdims=True)
rollout_valuezeroing_scores = compute_joint_attention(valuezeroing_scores, res=False)

cmap = "Blues"
all_tokens = [tokenizer.convert_ids_to_tokens(t) for t in inputs['input_ids']]
LAYERS = list(range(12))

# Increase figure size for more space
fig, axs = plt.subplots(3, 4, figsize=(70, 70))

# Increase hspace and wspace for more distance between subplots
plt.subplots_adjust(hspace=0.5, wspace=0.5)

for layer in LAYERS:
    a = layer // 4
    b = layer % 4

    sns.heatmap(ax=axs[a, b],
                data=pd.DataFrame(rollout_valuezeroing_scores[layer], index=all_tokens, columns=all_tokens), cmap=cmap,
                annot=False, cbar=False, xticklabels=True, yticklabels=True)
    axs[a, b].set_title(f"Layer: {layer + 1}")

    # Reduce font size to ensure text fits
    axs[a, b].set_xticklabels(axs[a, b].get_xticklabels(), fontsize=6)
    axs[a, b].set_yticklabels(axs[a, b].get_yticklabels(), fontsize=6)
    axs[a, b].set_xlabel('')
    axs[a, b].set_ylabel('')

# Ensure the layout is tight to further avoid overlaps
# fig.tight_layout()
print(all_tokens)
plt.show()

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
