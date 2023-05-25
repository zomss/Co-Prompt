import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn as nn
import os
from torch.nn.utils.rnn import pad_sequence

class RerankDataset(Dataset):
    def __init__(self, corpus, queries, qrels):
        super().__init__()
        self.documents, self.queries, self.labels = [],[],[]
        for q in qrels.keys():
            for d in qrels[q].keys():
                self.queries.append(queries[q])
                self.documents.append(corpus[d]['text'])
                self.labels.append((q,d))

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, i):
        return self.documents[i], self.queries[i], self.labels[i]


def get_model_from_huggingface(model):
    if "opt" in model:
        return AutoModelForCausalLM.from_pretrained(model), AutoTokenizer.from_pretrained(model), 900, True
    if "t5" in model:
        return AutoModelForSeq2SeqLM.from_pretrained(model), AutoTokenizer.from_pretrained(model), 480, False
    if "T0" in model:
        return T5ForConditionalGeneration.from_pretrained(model), T5Tokenizer.from_pretrained(model), 480, False
    raise Exception("Not supported model for Co-Prompt. Will be supported soon")

class Reranker(pl.LightningModule):

    def __init__(self, args, prompt):
        super().__init__()
        self.args = args
        self.model, self.tokenizer, self.doc_length, self.de = get_model_from_huggingface(args.hf_model_name)
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id

        self.template = args.template
        self.prompt = prompt
        self.delimiter = args.delimiter

        self.bs = args.batch_size

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def predict_step(self, batch, batch_idx):
        documents, queries, pairs = batch[0], batch[1], batch[2]
        bz = len(documents)
        if self.de:
            return self._de_predict(documents, queries, pairs, bz)
        else:
            return self._en2de_predict(documents, queries, pairs, bz)

    def _check_length(self, documents):
        documents = list(documents)
        for i,d in enumerate(documents):
            if len(self.tokenizer.tokenize(d)) > self.document_length:
                documents[i] = self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(d)[:self.doc_length])
        return tuple(documents)

    def _de_predict(self, documents, queries, pairs, bz):
        #Tokenization
        input = [
            [self.tokenizer.bos_token_id]
            + self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(self.template.format(d=d, v=self.delimiter,p=self.prompt).rstrip(" ")))
            for d in documents
        ]
        label_mask = torch.LongTensor([[len(i)] for i in input]).to(self.device)
        input = [
            torch.LongTensor(i + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(q))).to(self.device)
            for i, q in zip(input, queries)
        ]
        total_len = torch.LongTensor([[len(i)] for i in input]).to(self.device)
        input_ids = pad_sequence(input, True, padding_value=self.pad_token_id).long().to(self.device)
        attention_mask = input_ids != self.pad_token_id
        labels = torch.empty_like(input_ids).fill_(-100).long().to(self.device)

        label_ids = [
            torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(q))).to(self.device)
            for q in queries
        ]

        for bidx in range(bz):
            for i in range(len(label_ids[bidx])):
                labels[bidx][label_mask[bidx][0] + i] = label_ids[bidx][i]

        #Calculate Query Reconstruction Score
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        score = []
        for i in range(bz):
            log_softmax = nn.functional.log_softmax(logits[i][label_mask[i][0] - 1:total_len[i][0] - 1])
            nll = -log_softmax.gather(1, label_ids[i].unsqueeze(0).transpose(0, 1))
            avg_nll = torch.sum(nll, dim=0)
            score.append(float(-avg_nll) / float(total_len[i] - label_mask[i]))

        result = [((pairs[0][i], pairs[1][i]),score[i]) for i in range(len(score))]
        return result




    def _en2de_predict(self, documents, queries, pairs, bz):
        #Tokenization
        input = [
            torch.LongTensor(self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(self.template.format(d=d, v=self.delimiter, p=self.prompt).rstrip()))
                             + [self.tokenizer.eos_token_id]).to(self.device)
            for d in documents
        ]
        input_ids = pad_sequence(input, True, padding_value=self.pad_token_id).long().to(self.device)
        attention_mask = input_ids != self.pad_token_id
        labels = [
            torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(q.strip()))
                             + [self.tokenizer.eos_token_id]).to(self.device)
            for q in queries
        ]
        total_len = torch.LongTensor([[len(i)] for i in labels]).to(self.device)
        labels = pad_sequence(labels, True, padding_value=self.pad_token_id).long().to(self.device)

        #Calculate Query Reconstruction Score
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        label_ids = [
            torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(q))).to(self.device)
            for q in queries
        ]
        score = []
        for i in range(bz):
            log_softmax = nn.functional.log_softmax(logits[i])
            nll = -log_softmax.gather(1, label_ids[i].unsqueeze(0).transpose(0, 1))
            avg_nll = torch.sum(nll, dim=0)
            score.append(float(-avg_nll) / float(total_len[i]))

        result = [((pairs[0][i], pairs[1][i]),score[i]) for i in range(len(score))]
        return result


    def get_dataloader(self, corpus, queries, result):
        dataset = RerankDataset(corpus=corpus, queries=queries, qrels=result)
        return DataLoader(dataset, batch_size=self.bs, num_workers=5, persistent_workers=True)