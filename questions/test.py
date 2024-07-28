import json
from datasets import Dataset, DatasetDict
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import evaluate

# Загрузка кастомного датасета
with open("../dataset/ds.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Преобразование данных в формат datasets.Dataset
dataset = Dataset.from_list(data)
dataset = DatasetDict({"train": dataset, "validation": dataset})

# Загрузка модели и токенизатора T5
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Функция для подготовки данных
def preprocess_data(examples):
    inputs = ["question: " + question + " context: " + context for question, context in zip(examples["question"], examples["context"])]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # Настройка меток (labels)
    with tokenizer.as_target_tokenizer():
        labels = [ans['text'][0] if len(ans['text']) > 0 else '' for ans in examples["answers"]]
        labels = tokenizer(labels, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Применение функции подготовки данных к датасету
tokenized_datasets = dataset.map(preprocess_data, batched=True, remove_columns=["context", "question", "answers"])

# Настройка параметров тренировки
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    save_steps=500,
)

# Определение метрики для оценки качества
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(dim=-1) if isinstance(predictions, tuple) else predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(pred.split()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.split()) for label in decoded_labels]

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {key: value.mid.fmeasure * 100 for key, value in result.items()}

# Создание Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Запуск обучения
trainer.train()
