import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn as nn
import os
from torch.nn.utils.rnn import pad_sequence

def get_model_from_huggingface(model):
    if "opt" in model:
        return AutoModelForCausalLM.from_pretrained(model), AutoTokenizer.from_pretrained(model), 900
    if "t5" in model:
        return AutoModelForSeq2SeqLM.from_pretrained(model), AutoTokenizer.from_pretrained(model), 480
    if "T0" in model:
        return T5ForConditionalGeneration.from_pretrained(model), T5Tokenizer.from_pretrained(model), 480
    raise Exception("Not supported model for Co-Prompt. Will be supported soon")

class Reranker(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model, self.tokenizer, self.doc_length = get_model_from_huggingface(args.model_name)
