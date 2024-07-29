from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Загрузка обученной модели и токенизатора
model_path = "../model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to('cuda')

def answer_question(question):
    input_text = f"question: {question}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to('cuda')
    outputs = model.generate(**inputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Примеры использования
questions = [
    "What is the capital of France?",
    "What is a famous landmark in Paris?",
    "Who is Putin?"
]

# Получение и вывод ответов
for question in questions:
    answer = answer_question(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}\n")
