import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq

# Загрузка данных из JSON-файла
with open('../dataset/ds.json', 'r') as f:
    data = json.load(f)

# Преобразование данных в DataFrame и затем в Dataset
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# Настройка токенизатора и модели
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = [f"context: {c} question: {q}" for c, q in zip(examples["context"], examples["question"])]
    targets = examples["answer"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(targets, max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Настройки тренировки
training_args = Seq2SeqTrainingArguments(
    output_dir="../results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=500,
    weight_decay=0.01,
    save_total_limit=500,
    predict_with_generate=True,
    logging_dir="../logs",
    logging_steps=10,
    fp16=True,
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Запуск обучения
trainer.train()

# Сохранение модели и токенизатора
model.save_pretrained("../model")
tokenizer.save_pretrained("../model")
