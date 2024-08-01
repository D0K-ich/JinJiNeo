import pandas as pd
import torch
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, \
    T5Tokenizer, T5ForConditionalGeneration

data = pd.read_csv('../dataset/ds.csv')
data['text'] = data['question'] + " " + data['answer']
dataset = Dataset.from_pandas(data[['text']])

model_name = "../model"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    texts = examples['text']

    if isinstance(texts, list):
        texts = [str(text) for text in texts]
    else:
        texts = [str(texts)]

    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="../results",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="../logs",
    logging_steps=2,
)

# Создание тренера
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

trainer.train()

trainer.save_model("../model-trained")
tokenizer.save_pretrained("../model-trained")
