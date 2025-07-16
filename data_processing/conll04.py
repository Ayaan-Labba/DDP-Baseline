import sys
sys.path.append("./data_processing/")

from common import get_pred_set

def get_conll04_labels(dataset):
    labels = set()
    for example in dataset:
        for rel in example["relations"]:
            labels.add(rel["relation_text"])
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

        pred = set()
        for rel in preds:
            if rel["score"] < threshold:
                continue
            pred.add((tuple(rel["head_pos"]), tuple(rel["tail_pos"]), rel["label"]))

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