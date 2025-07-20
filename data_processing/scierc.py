import sys
sys.path.append("./data_processing/")

from common import get_pred_set

def get_scierc_labels(dataset):
    labels = set()
    for example in dataset:
        for rels in example["relations"]:
            for _, _, _, _, label in rels:
                labels.add(label)
    return sorted(list(labels))

def create_scierc_input(example):
    sentences = example["sentences"]
    ner_per_sent = example["ner"]

    tokens = []
    ner_output = []
    offset = 0

    for sent, ner_spans in zip(sentences, ner_per_sent):
        tokens.extend(sent)
        for span in ner_spans:
            start, end, label = span
            entity_text = " ".join(sent[start-offset:end + 1-offset])
            ner_output.append([start, end, label, entity_text])
        offset += len(sent)

    return {
        "tokens": tokens,
        "ner": ner_output
    }

def get_scierc_gold(example):
    gold = set()
    for rels in example['relations']:
        for rel in rels:
            head_start, head_end, tail_start, tail_end, label = rel
            head_span = (head_start, head_end + 1)
            tail_span = (tail_start, tail_end + 1)
            gold.add((head_span, tail_span, label))
    return gold

def evaluate_scierc(dataset, predictions, threshold=0.5):
    assert len(dataset) == len(predictions)

    tp = fp = fn = 0

    for example, preds in zip(dataset, predictions):
        gold = get_scierc_gold(example)
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