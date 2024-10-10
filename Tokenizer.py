from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import openai

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
