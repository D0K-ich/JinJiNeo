import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq

# Загрузка данных из JSON-файла
with open('../dataset/ds.json', 'r') as f:
    data = json.load(f)

# Преобразование данных в DataFrame, а затем в Dataset
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# Токенизация данных
model_name = "../model"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = [f"question: {q}" for q in examples["question"]]
    targets = [a for a in examples["answer"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(targets, max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Настройки тренировки с интеграцией TensorBoard
training_args = Seq2SeqTrainingArguments(
    output_dir="../model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # Уменьшено для примера
    per_device_eval_batch_size=16,
    num_train_epochs=300,
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,
    logging_dir="../logs",  # Директория для логов TensorBoard
    logging_steps=10,      # Частота логирования
    fp16=True,            # Использование 16-битной арифметики (если поддерживается)
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
