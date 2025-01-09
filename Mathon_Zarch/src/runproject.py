import time
from statistics import mean

import pandas as pd
from pprint import pprint
import pyrallis
from lightning import seed_everything

from config import Config
from classifier_wrapper import ClassifierWrapper


def load_data():
    df_train = pd.read_csv("../data/ftdataset_train.tsv", sep=' *\t *', encoding='utf-8', engine='python')
    df_val = pd.read_csv("../data/ftdataset_val.tsv", sep=' *\t *', encoding='utf-8', engine='python')
    df_test = pd.read_csv("../data/ftdataset_test.tsv", sep=' *\t *', encoding='utf-8', engine='python')
    return df_train.to_dict(orient='records'), df_val.to_dict(orient='records'), df_test.to_dict(orient='records')

def eval(preds: list[dict], test_data: list[dict]) -> dict[str,float]:
    n = len(test_data)
    aspects = {'Prix', 'Cuisine', 'Service', 'Ambiance'}
    correct_counts = {aspect: 0.0 for aspect in aspects}
    for pred, ref in zip(preds, test_data):
        if pred is None:
            continue
        for aspect in aspects:
            if pred[aspect] == ref[aspect]:
                correct_counts[aspect] += 1
    for aspect in correct_counts:
        correct_counts[aspect] = round(100*correct_counts[aspect]/n, 2)
    macro_acc = round(sum(acc for acc in correct_counts.values())/len(aspects), 2)
    correct_counts['macro_acc'] = macro_acc
    return correct_counts

def run_project(cfg: Config):
    train_data, val_data, test_data = load_data()
    if cfg.n_train > 0:
        train_data = train_data[:cfg.n_train]
    if cfg.n_test > 0:
        test_data = test_data[:cfg.n_test]
    test_texts = [element['Avis'] for element in test_data]
    if ClassifierWrapper.METHOD == 'LLM':
        cfg.n_runs = 1
    all_runs_acc = []
    for run_id in range(1, cfg.n_runs+1):
        print(f"RUN {run_id}/{cfg.n_runs}")
        clwrapper = ClassifierWrapper(cfg)
        if ClassifierWrapper.METHOD == 'PLMFT':
            print("Training...")
            clwrapper.train(train_data, val_data, cfg.device)
        print("Evaluation on test split...")
        preds = clwrapper.predict(test_texts, cfg.device)
        accuracies = eval(preds, test_data)
        all_runs_acc.append(accuracies['macro_acc'])
    print("ALL RUNS ACC:", all_runs_acc)
    avg_acc = round(mean(all_runs_acc), 2)
    print("AVG MACRO ACC:", avg_acc)

if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=Config)
    pprint(vars(cfg), sort_dicts=False, compact=True)
    seed_everything(123)
    start_time = time.perf_counter()
    run_project(cfg)
    total_exec_time = round(time.perf_counter() - start_time, 1)
    print("TOTAL EXEC TIME:", total_exec_time)
