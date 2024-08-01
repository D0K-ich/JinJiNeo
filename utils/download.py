import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForQuestionAnswering, AutoTokenizer, \
    GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM

from datasets import load_dataset, load, Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
    losses,
    evaluation
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
import pandas as pd

model_name = "google/flan-t5-large"  # Можно использовать "t5-base" или "t5-large" для более мощных моделей
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

model.save_pretrained("../model")
tokenizer.save_pretrained("../model")
