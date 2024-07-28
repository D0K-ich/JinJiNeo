import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

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

dataset = load_dataset("squad")
dataset.save_to_disk("../dataset")
