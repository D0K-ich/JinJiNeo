from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch

# Загрузка предобученной модели и токенизатора
model_name = "../model-trained"  # Можно использовать "t5-base" или "t5-large" для более мощных моделей
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Использование CUDA, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def generate_answer(question, max_length=128, num_beams=3):

    input_text = f"question: {question} context: "
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    try:
        # Генерация ответа
        outputs = model.generate(input_ids, max_length=max_length, num_beams=num_beams, early_stopping=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Пример использования функции с несколькими вопросами
questions = [
    "What is the capital of France?",
    "Who wrote 'Hamlet'?",
    "What is the speed of light?",
    "Who is Dok Sanches?",
]

for question in questions:
    answer = generate_answer(question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")