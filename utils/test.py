from transformers import AutoModelForCausalLM, AutoTokenizer

# Загрузка модели и токенизатора
model_name = '../model'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Функция для генерации ответа на вопрос
tokenizer.pad_token = tokenizer.eos_token

# Функция для генерации ответа на вопрос
def generate_answer(question):
    inputs = tokenizer.encode(question, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        num_beams=5,
        top_p=0.95,
        top_k=50
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

question = "What is the capital of France?"
answer = generate_answer(question)
print(f"Answer: {answer}")