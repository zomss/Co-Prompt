import argparse, json, time, os
from src import generator, reranker, util, retrieve
from beir import LoggingHandler
import logging, pathlib
import pytorch_lightning as pl
import ast

os.environ['WANDB_MODE'] = 'offline'
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def timestr():
    return time.strftime("%Y%m%d-%H%M%S")

def parse():
    parser = argparse.ArgumentParser()

    # GPU Setting
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=2)

    # Dataset
    parser.add_argument("--dataset", type=str, default="msmarco")
    parser.add_argument("--search_split", type=str, default="test")
    parser.add_argument("--eval_split", type=str, default="dev")

    parser.add_argument("--retriever", type=str, default="BM25")

    #Reranker
    parser.add_argument("--hf_model_name", type=str, default="../models/opt-2.7b")
    parser.add_argument("--prompt", type=str, default="Please write a question based on this passage.")
    parser.add_argument("--delimiter", type=str, default="\n")
    parser.add_argument("--template", type=str, default="Passage: {d}{v}{p}{v}")

    ##Prompt Search
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--generator",type=str, default="../models/opt-2.7b")
    parser.add_argument("--beam", type=int, default=10)
    parser.add_argument("--length", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--neg", action="store_true")
    parser.add_argument("--start_token", type=str, default="Please")
    parser.add_argument("--prompt_no", type=int, default=0)
    parser.add_argument("--sample_size", type=int, default=1500)

    # dir
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--dataset_dir", type=str, default="./data")
    parser.add_argument("--result_dir", type=str, default="./result/{t}")
    parser.add_argument("--save_file_name", type=str, default="{r}_{s}.json")
    parser.add_argument("--raw_result_file", type=str, default="raw_result.json")
    parser.add_argument("--metric_result_file", type=str, default="metric_result.json")
    parser.add_argument("--prompt_dir", type=str, default="./prompts/{d}/")
    parser.add_argument("--prompt_file", type=str, default="model_{m}_beam_{b}_length_{l}_top_{t}_neg_{n}_start_{s}.json")

    return parser.parse_args()

def load_retrieve(args, corpus, queries):
    result_path = os.path.join(args.dataset_dir, args.dataset)
    result_file = os.path.join(result_path, args.save_file_name.format(r=args.retriever, s=args.eval_split))
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            results = json.load(f)
        return results
    else:
        return retrieve.retrieve(args, corpus, queries)

def rerank(args, corpus, queries, result, prompt, score=False):
    writer = util.CustomWriter(os.path.join(pathlib.Path(__file__).parent.absolute(), "result"))
    trainer = pl.Trainer(accelerator="gpu", devices=args.gpus, strategy="ddp_spawn",
                         callbacks=writer)
    model = reranker.Reranker(args, prompt)
    dataloader = model.get_dataloader(corpus, queries, result)
    trainer.predict(model, dataloaders=dataloader)
    if not score:
        reranked_results = writer.get_data_from_files(trainer, result)
        del trainer
        return reranked_results
    else:
        score = float(writer.get_scores_from_files(trainer))
        del trainer
        return score

def search(args):
    args.prompt_dir = args.prompt_dir.format(d=args.dataset)
    if not os.path.exists(args.prompt_dir):
        os.makedirs(args.prompt_dir, exist_ok=True)
    prompt_file = os.path.join(args.prompt_dir, args.prompt_file.format(
        m=args.hf_model_name.split("/")[1], b=args.beam, l=args.length, t=args.top_k, n=args.neg, s=args.start_token
    ))

    if not os.path.exists(prompt_file):
        corpus, queries, qrels, _ = util.load_data(args.dataset, args.dataset_dir, args.search_split)
        pos_qrels, _ = util.get_qrels(args, qrels)
        gen_model = generator.generator(args)

        total_prompts = dict()
        gen_prompts = {args.start_token : 0}
        for _ in range(args.length):
            gen_prompts = {p : rerank(args, corpus, queries, pos_qrels, p, True) for prompt in gen_prompts.keys() for p in gen_model.get_tokens(prompt) }
            gen_prompts = sorted(gen_prompts.items(), key=lambda item: item[1], reverse=True)
            gen_prompts = {gen_prompts[i][0]: gen_prompts[i][1] for i in range(args.top_k)}
            total_prompts.update(gen_prompts)

        with open(prompt_file, 'w') as fw:
            p = {'prompts': total_prompts}
            json.dump(p, fw)

    with open(prompt_file, 'r') as fr:
        prompts = json.load(fr)['prompts']

    return prompts[args.prompt_no]


def main():
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    t = timestr()
    args = parse()
    logging.info("The stored time is {}".format(t))

    if args.search:
        prompt = search(args)
    else:
        prompt = args.prompt

    test_corpus, test_queries, test_qrels, data_path = util.load_data(args.dataset, args.dataset_dir,
                                                                      args.eval_split)
    test_results = load_retrieve(args, test_corpus, test_queries)
    util.evaluate_result(test_results, test_qrels)

    reranked_results = rerank(args, test_corpus, test_queries, test_results, prompt)

    args.result_dir = args.result_dir.format(t=t)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir, exist_ok=True)
    with open(os.path.join(args.result_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f)

    ndcg, _map, recall, precision, top_k = util.evaluate_result(reranked_results, test_qrels)
    util.record_metric(ndcg, _map, recall, precision,top_k, out_dir=args.result_dir, prompt=prompt,
                           out_file=args.metric_result_file)

if __name__=="__main__":
    main()
