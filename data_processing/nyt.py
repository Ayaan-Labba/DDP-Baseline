from common import get_pred_set

def get_nyt_labels(dataset):
    labels = set()
    for example in dataset:
        for rel in example["relations"]:
            labels.add(rel["relation_text"])
    
    return sorted(list(labels))

def create_nyt_input(example):
    tokens = example["tokenized_text"]
    ner = [[span[0], span[1]-1, span[2], span[3]] for span in example["ner"]]

    return {
        "tokens": tokens,
        "ner": ner
    }

def get_nyt_gold_set(example):
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
        gold = get_nyt_gold_set(example)
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