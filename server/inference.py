import transformers.tokenization_utils_base
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from server.djinji import DjinModel

tokenizer   = transformers.models.auto.AutoTokenizer
model       = transformers.models.auto.auto_factory._BaseAutoModelClass

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name:str, max_lengh:int=128, num_beams:int=3, temp:float=0.7):
    global tokenizer, model
    tokenizer   = AutoTokenizer.from_pretrained(model_name)
    model       = AutoModelForCausalLM.from_pretrained(model_name)
    model       = model.to(device)

    return DjinModel(max_lengh, temp, num_beams, model, tokenizer)