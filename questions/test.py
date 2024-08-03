from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained("../model")
model = AutoModelForCausalLM.from_pretrained("../model")
input_text = "Какая столица России?"

# Токенизация ввода
inputs = tokenizer(input_text, return_tensors="pt")

# Генерация ответа
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=256,  # Максимальная длина генерируемого текста
        temperature=0.7,  # Управляет креативностью ответа
        top_p=0.9,  # Управляет сэмплированием топ-выборок
        do_sample=True,  # Включает сэмплирование
    )

# Раскодирование ответа модели обратно в текст
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Вывод результата
print(generated_text)