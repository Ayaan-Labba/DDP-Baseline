from nltk.tokenize.treebank import TreebankWordDetokenizer

import sys
sys.path.append("./data_processing/")

from common import get_pred_set

def get_nyt_labels(dataset):
    labels = set()
    for example in dataset:
        for rel in example["relations"]:
            labels.add(rel["relation_text"])
    
    return sorted(list(labels))

def create_nyt_input(example):
    tokens = example["tokenized_text"]
    
    seen_spans = set()
    unique_ner = []
    
    for span in example["ner"]:
        start = span[0]
        end = span[1] - 1 # Adjust end to be inclusive
        label = span[2]
        text = span[3]
        
        if (start, end) not in seen_spans:
            seen_spans.add((start, end))
            unique_ner.append([start, end, label, text])
    
    return {
        "tokens": tokens,
        "ner": unique_ner
    }

def get_nyt_gold(example):
    gold = set()
    for rel in example['relations']:
        head = rel["head"]["position"]
        tail = rel["tail"]["position"]
        relation = rel["relation_text"]

        head_span = (head[0], head[1])
        tail_span = (tail[0], tail[1])

        gold.add((head_span, tail_span, relation))
    
    return gold

def evaluate_nyt(dataset, predictions, threshold=0.5):
    assert len(dataset) == len(predictions)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    for example, preds in zip(dataset, predictions):
        gold = get_nyt_gold(example)
        pred = get_pred_set(preds, threshold)
        
        # for rel in gold:
        #     rel = (rel[0], rel[1], rel[2].lower().strip())
        # for rel in pred:
        #     rel = (rel[0], rel[1], rel[2].lower().strip())

        tp = len(gold & pred)
        fp = len(pred - gold)
        fn = len(gold - pred)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn
    }

def get_nyt_errors(dataset, predictions, threshold=0., top_k=3):
    assert len(dataset) == len(predictions)

    detokenizer = TreebankWordDetokenizer()
    error_report = []

    for example, preds in zip(dataset, predictions):
        gold_set = get_nyt_gold(example)
        pred_set = get_pred_set(preds, threshold)
        
        # Reconstruct text by joining tokens
        tokens = example['tokenized_text']
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

def get_top_nyt_errors(error_report, top_n=10):
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