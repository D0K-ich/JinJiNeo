from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained("../model")
tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)

def generate_answer(question, context, max_length=50, min_length=10):
    # Форматирование ввода для T5
    input_text = f"question: {question}  context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Генерация ответа
    output_ids = model.generate(input_ids, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

# Пример использования
context = """
Франция — это страна в Западной Европе. Её столица — Париж, который известен своими музеями и архитектурными памятниками.
"""
question = "Hello!"

# Задать max_length для увеличения длины ответа
answer = generate_answer(question, context, max_length=512, min_length=10)
print(f"Вопрос: {question}")
print(f"Ответ: {answer}")
