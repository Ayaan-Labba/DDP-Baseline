from nltk.tokenize.treebank import TreebankWordDetokenizer
from typing import Dict # type: ignore

import sys
sys.path.append("./data_processing/")

from common import get_relation_tuples

def process_conll04(example: Dict) -> Dict:
    """ Process a single CoNLL-04 data point into GLiREL acceptable format.
    Args:
        example (Dict): A CoNLL-04 data point; should include 'tokens', 'entities', and 'relations' keys.
    Returns:
        Dict: Processed data point in GLiREL acceptable format with 'text', 'tokens', 'ner', and 'gold_relations' keys.
    """
    tokens = example['tokens']

    detokenizer = TreebankWordDetokenizer()
    # Reconstruct text by joining tokens
    text = detokenizer.detokenize(tokens)

    # Convert entities to GLiREL NER format
    ner = []
    for ent in example['entities']:
        start = ent['start']
        end_inclusive = ent['end'] - 1  # GLiREL expects inclusive end
        ent_text = detokenizer.detokenize(tokens[start:ent['end']])
        ner.append([start, end_inclusive, ent['type'], ent_text])

    # Convert relations to GLiREL output format (exclusive end indices)
    relations = []
    for rel in example['relations']:
        head_ent = example['entities'][rel['head']]
        tail_ent = example['entities'][rel['tail']]
        relations.append({
            'head_pos': [head_ent['start'], head_ent['end']],
            'tail_pos': [tail_ent['start'], tail_ent['end']],
            'head_text': detokenizer.detokenize(tokens[head_ent['start']:head_ent['end']]),
            'tail_text': detokenizer.detokenize(tokens[tail_ent['start']:tail_ent['end']]),
            'label': rel['type'],
            'score': 1.0
        })

    return {
        'text': text,
        'tokens': tokens,
        'ner': ner,
        'relations': relations
    }

def process_conll04_with_new_labels(example: Dict, label_map) -> Dict:
    """ Process a single CoNLL-04 data point into GLiREL acceptable format and with new labels.
    Args:
        example (Dict): A CoNLL-04 data point; should include 'tokens', 'entities', and 'relations' keys.
    Returns:
        Dict: Processed data point in GLiREL acceptable format with 'text', 'tokens', 'ner', and 'gold_relations' keys.
    """
    tokens = example['tokens']

    detokenizer = TreebankWordDetokenizer()
    # Reconstruct text by joining tokens
    text = detokenizer.detokenize(tokens)

    # Convert entities to GLiREL NER format
    ner = []
    for ent in example['entities']:
        start = ent['start']
        end_inclusive = ent['end'] - 1  # GLiREL expects inclusive end
        ent_text = detokenizer.detokenize(tokens[start:ent['end']])
        ner.append([start, end_inclusive, ent['type'], ent_text])

    # Convert relations to GLiREL output format (exclusive end indices)
    relations = []
    for rel in example['relations']:
        label = label_map[rel['type']]

        head_ent = example['entities'][rel['head']]
        tail_ent = example['entities'][rel['tail']]
        relations.append({
            'head_pos': [head_ent['start'], head_ent['end']],
            'tail_pos': [tail_ent['start'], tail_ent['end']],
            'head_text': tokens[head_ent['start']:head_ent['end']],
            'tail_text': tokens[tail_ent['start']:tail_ent['end']],
            'label': label,
            'score': 1.0
        })

    return {
        'text': text,
        'tokens': tokens,
        'ner': ner,
        'relations': relations
    }

def fuzzy_evaluate_conll04(dataset, predictions, threshold=0.5, top_k=1):
    assert len(dataset) == len(predictions)

    tp = fp = fn = 0

    for example, preds in zip(dataset, predictions):
        gold = get_relation_tuples(example['relations'])
        gold_set = set()
        for rel in gold:
            if rel[2] == "OrgBased_In":
                label = "Located_In"
            elif rel[2] == "organisation is based in":
                label = "is located in"
            else:
                label = rel[2]
            # label = rel[2]
            # if label == 'OrgBased_In':
            #     gold_set.add((tuple(rel[0]), tuple(rel[1]), 'Located_In'))
            # elif label == 'organisation is based in':
            #     gold_set.add((tuple(rel[0]), tuple(rel[1]), 'is located in'))
            gold_set.add((tuple(rel[0]), tuple(rel[1]), label))

        pred = get_relation_tuples(preds, threshold, top_k)
        pred_set = set()
        for rel in pred:
            if rel[2] == "OrgBased_In":
                label = "Located_In"
            elif rel[2] == "organisation is based in":
                label = "is located in"
            else:
                label = rel[2]
            pred_set.add((tuple(rel[0]), tuple(rel[1]), label))

        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'TP': tp,
        'FP': fp,
        'FN': fn
    }