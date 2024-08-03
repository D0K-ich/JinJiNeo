import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForQuestionAnswering, AutoTokenizer, \
    GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM, AutoModelForCausalLM

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

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", use_auth_token=True)

model.save_pretrained("../gemma")
tokenizer.save_pretrained("../gemma")
