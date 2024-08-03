import torch
import transformers

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DjinModel:
    def __init__(self, max_length:int, temp:float, num_beams:int, model:transformers.modeling_utils.PreTrainedModel, tokenizer:transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        self.model      = model
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.num_beams  = num_beams
        self.temp       = temp


    def generate_answer(self, question):
        print(f"Start generate answer for question : {question}")

        input_ids = self.tokenizer(question, return_tensors="pt").input_ids.to(device)

        try:
            outputs = self.model.generate(input_ids, max_length=self.max_length, num_beams=self.num_beams, early_stopping=True, temperature=self.temp, top_p=0.9, do_sample=True)
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"End generate answer is {answer}")
            return answer
        except Exception as e:
            return f"An error occurred: {str(e)}"