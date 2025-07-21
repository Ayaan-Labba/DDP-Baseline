from nltk.tokenize.treebank import TreebankWordDetokenizer

import sys
sys.path.append("./data_processing/")

from common import get_pred_set

def get_conll04_labels(dataset):
    labels = set()
    for example in dataset:
        for rel in example["relations"]:
            labels.add(rel["type"])
    return sorted(list(labels))

def create_conll04_input(example):
    tokens = example["tokens"]
    ner = []
    for ent in example['entities']:
        start = ent['start']
        end = ent['end'] - 1  # make inclusive
        label = ent['type']
        text = ' '.join(tokens[start:end + 1])
        ner.append([start, end, label, text])

    return {
        "tokens": tokens,
        "ner": ner,
    }

def get_conll04_gold(example):
    gold = set()
    for rel in example['relations']:
        head = [example['entities'][rel["head"]]['start'], example['entities'][rel["head"]]['end']]
        tail = [example['entities'][rel["tail"]]['start'], example['entities'][rel["tail"]]['end']]
        relation = rel['type']

        head_span = (head[0], head[1])
        tail_span = (tail[0], tail[1])

        gold.add((head_span, tail_span, relation))

    return gold

def evaluate_conll04(dataset, predictions, threshold=0.5):
    assert len(dataset) == len(predictions)

    tp = fp = fn = 0

    for example, preds in zip(dataset, predictions):
        gold = get_conll04_gold(example)
        pred = get_pred_set(preds, threshold)

        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "TP": tp,
        "FP": fp,
        "FN": fn
    }

def fuzzy_evaluate_conll04(dataset, predictions, threshold=0.5):
    assert len(dataset) == len(predictions)

    tp = fp = fn = 0

    for example, preds in zip(dataset, predictions):
        gold = get_conll04_gold(example)
        gold_set = set()
        for rel in gold:
            label = 'Located_In' if rel[2] == 'OrgBased_In' else rel[2]
            gold_set.add((tuple(rel[0]), tuple(rel[1]), label))

        pred = get_pred_set(preds, threshold)
        pred_set = set()
        for rel in pred:
            label = 'Located_In' if rel[2] == 'OrgBased_In' else rel[2]
            pred_set.add((tuple(rel[0]), tuple(rel[1]), label))

        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "TP": tp,
        "FP": fp,
        "FN": fn
    }

def get_conll04_errors(dataset, predictions, threshold=0., top_k=3):
    assert len(dataset) == len(predictions)

    detokenizer = TreebankWordDetokenizer()
    error_report = []

    for example, preds in zip(dataset, predictions):
        gold_set = get_conll04_gold(example)
        pred_set = get_pred_set(preds, threshold)
        
        # Reconstruct text by joining tokens
        tokens = example['tokens']
        text = detokenizer.detokenize(tokens)
        
        def enrich_relation(rel):
            (h_start, h_end), (t_start, t_end), label = rel
            return ((h_start, h_end, tokens[h_start:h_end]), (t_start, t_end, tokens[t_start:t_end]), label)

        gold_relations = [enrich_relation(r) for r in gold_set]

        example_fp = []
        example_fn = []

        # Sort all predictions for top-K
        top_preds = sorted(preds, key=lambda x: -x["score"])[:top_k]
        enriched_top_preds = []
        for rel in top_preds:
            key = (tuple(rel["head_pos"]), tuple(rel["tail_pos"]), rel["label"])
            enriched_top_preds.append({
                "relation": enrich_relation(key),
                "score": rel["score"]
            })

        # False Positives
        for rel in preds:
            key = (tuple(rel["head_pos"]), tuple(rel["tail_pos"]), rel["label"])
            if key not in gold_set and rel["score"] >= threshold:
                example_fp.append({
                    "relation": enrich_relation(key),
                    "score": rel["score"]
                })

        # False Negatives
        for key in gold_set:
            if key not in pred_set:
                score = 0.0
                for rel in preds:
                    pred_key = (tuple(rel["head_pos"]), tuple(rel["tail_pos"]), rel["label"])
                    if pred_key == key:
                        score = rel["score"]
                        break
                example_fn.append({
                    "relation": enrich_relation(key),
                    "score": score,
                    "top_predictions": enriched_top_preds
                })

        # Sort by score
        example_fp.sort(key=lambda x: -x["score"])
        example_fn.sort(key=lambda x: x["score"])

        if example_fp or example_fn:
            error_report.append({
                "text": text,
                "gold": gold_relations,
                "false_positives": example_fp,
                "false_negatives": example_fn
            })

    return error_report

def get_top_conll04_errors(error_report, top_n=10):
    top_fp_examples = []
    top_fn_examples = []

    for entry in error_report:
        text = entry["text"]
        gold = entry["gold"]

        if entry["false_positives"]:
            top_fp = entry["false_positives"][0]
            top_fp_examples.append({
                "text": text,
                "gold": gold,
                "relation": top_fp["relation"],
                "score": top_fp["score"]
            })

        if entry["false_negatives"]:
            top_fn = entry["false_negatives"][0]
            top_fn_examples.append({
                "text": text,
                "gold": gold,
                "relation": top_fn["relation"],
                "score": top_fn["score"],
                "top_predictions": top_fn["top_predictions"]
            })

    # Sort overall
    sorted_fp = sorted(top_fp_examples, key=lambda x: -x["score"])[:top_n]
    sorted_fn = sorted(top_fn_examples, key=lambda x: x["score"])[:top_n]

    return {
        "top_false_positives": sorted_fp,
        "top_false_negatives": sorted_fn
    }