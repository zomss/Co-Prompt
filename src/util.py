from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.util import download_and_unzip
from pytorch_lightning.callbacks import BasePredictionWriter
from typing import Any, List, Sequence, Optional
import os, json
import logging

def load_data(dataset, dataset_dir, split):
    if os.path.exists(os.path.join(dataset_dir, dataset)):
        data_path = os.path.join(dataset_dir, dataset)
    else:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        data_path = download_and_unzip(url, dataset_dir)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
    return corpus, queries, qrels, data_path

def evaluate_result(results, qrels):
    retriever = EvaluateRetrieval()
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1, 3, 5, 10, 20, 50, 80, 100])
    top_k = retriever.evaluate_custom(qrels, results, [1,3,5,10,20,50,80,100], metric="top_k_acc")
    mrr = retriever.evaluate_custom(qrels, results, [1,3,5,10,20,50,80,100], metric="mrr")
    return ndcg, _map, recall, precision, top_k

def record_metric(ndcg, _map, recall, precision, top_k, prompt, out_dir, out_file):
    result = {
        "ncdg" : ndcg,
        "map" : _map,
        "recall" : recall,
        "precision" : precision,
        "top_k" : top_k,
        "prompt" : prompt
    }

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, out_file)
    with open(out_file, 'w') as f:
        json.dump(result, f)

def get_qrels(args, qrels, sample_size = -1):
    pos_qrels = {}
    p_len = 0

    for q in qrels.keys():
        for d in qrels[q].keys():
            if qrels[q][d] > 0:
                if q not in pos_qrels:
                    pos_qrels[q] = {}
                pos_qrels[q][d] = qrels[q][d]
                p_len += 1
            if sample_size > 0 and p_len >= sample_size:
                return pos_qrels
    return pos_qrels

def get_result(predictions):
    result = {}
    for predict in predictions:
        for label, score in predict:
            if label[0] not in result:
                result[label[0]] = {}
            if label[1] not in result[label[0]]:
                result[label[0]][label[1]] = []
            result[label[0]][label[1]].append(score)

    for q in result.keys():
        for d in result[q].keys():
            result[q][d] = sum(result[q][d])/len(result[q][d])
    return result

class CustomWriter(BasePredictionWriter):
    PRED_FILENAME_EXT = "json"

    def __init__(self, out_dir, wrtie_interval="epoch"):
        super().__init__(wrtie_interval)
        self.results = []
        self.out_dir = out_dir
        self.logger = logging.getLogger(type(self).__name__)

    def get_pred_file_name(self, global_rank=None):
        file_name = "pred_{}.{}".format(global_rank, self.PRED_FILENAME_EXT)
        return os.path.join(self.out_dir, file_name)

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.results = [[] for _ in range(trainer.world_size)]

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ):
        result = get_result(predictions)
        file_dir = self.get_pred_file_name(trainer.global_rank)
        with open(file_dir, 'w') as f:
            json.dump(result, f)

    def get_scores_from_files(self, trainer):
        score = []
        for i in range(trainer.world_size):
            file_dir = self.get_pred_file_name(i)
            with open(file_dir, 'r') as f:
                data = json.load(f)
            for q in data.keys():
                for d in data[q].keys():
                    score.append(data[q][d])
        return sum(score)/len(score)

    def get_data_from_files(self, trainer, result):
        for i in range(trainer.world_size):
            file_dir = self.get_pred_file_name(i)
            with open(file_dir, 'r') as f:
                data = json.load(f)
            for q in data.keys():
                for d in data[q].keys():
                    result[q][d] = data[q][d]
        return result
