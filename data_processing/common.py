import json
from typing import List, Dict # type: ignore

def load_json(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]
    
def save_json(data: List[Dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def get_relation_labels(dataset) -> List:
    labels = set()
    for example in dataset:
        for rel in example['relations']:
            labels.add(rel['label'])
    
    return sorted(list(labels))

def get_entity_types(dataset) -> List:
    ent_types = set()
    for example in dataset:
        for ent in example['ner']:
            ent_types.add(ent[2])

    return sorted(list(ent_types))

def get_relation_tuples(relations, threshold=0.5, top_k=1) -> set:
    rels = set()
    relations = sorted(relations, key=lambda x: -x['score'])
    
    if top_k == 1:
        rels.add((tuple(relations[0]['head_pos']), tuple(relations[0]['tail_pos']), relations[0]['label'])) if relations[0]['score'] >= threshold else rels
        for rel in relations:
            if rel['score'] < 1.0:
                continue
            rels.add((tuple(rel['head_pos']), tuple(rel['tail_pos']), rel['label']))
    else:
        for rel in relations:
            if rel['score'] < threshold:
                continue
            rels.add((tuple(rel['head_pos']), tuple(rel['tail_pos']), rel['label']))
    
    return rels

def run_inference(model, dataset, labels, threshold=0.5, top_k=1, batch_size=64) -> List[Dict]:
    print("Running inference on device:", model.device)
    
    batch_size = 64
    batches = [dataset[i:i+batch_size] for i in range(0,len(dataset),batch_size)]
    predictions = []

    for batch in batches:
        tokens = []
        ner = []
        
        for entry in batch:
            tokens.append(entry['tokens'])
            ner.append(entry['ner'])
        
        pred = model.batch_predict_relations(tokens, labels=labels, ner=ner, threshold=threshold, top_k=top_k)
        predictions.extend(pred)
    
    # predictions = []
    # for example in input:
    #     tokens = example['tokens']
    #     ner = example['ner']
    #     prediction = model.predict_relations(tokens, labels, threshold=threshold, ner=ner, top_k=top_k)
    #     predictions.append(prediction)
    
    return predictions

def evaluate(dataset, predictions, threshold=0.5, top_k=1) -> Dict:
    assert len(dataset) == len(predictions)

    tp = fp = fn = 0

    for example, preds in zip(dataset, predictions):
        gold = get_relation_tuples(example['relations'])
        pred = get_relation_tuples(preds, threshold, top_k=top_k)

        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1-score': round(f1, 4),
        'True_Positives': tp,
        'False_Positives': fp,
        'False_Negatives': fn
    }

def get_error_report(dataset, predictions, threshold=0., top_k=5) -> List[Dict]:
    assert len(dataset) == len(predictions)

    error_report = []

    for example, preds in zip(dataset, predictions):
        gold_set = get_relation_tuples(example['relations'])
        pred_set = get_relation_tuples(preds, threshold)
        tokens = example['tokens']
        text = example['text']
        
        def enrich_relation(rel):
            (h_start, h_end), (t_start, t_end), label = rel
            return ((h_start, h_end, ' '.join(tokens[h_start:h_end])), (t_start, t_end, ' '.join(tokens[t_start:t_end])), label)

        gold_relations = [enrich_relation(r) for r in gold_set]
        example_fp = []
        example_fn = []

        # Sort all predictions for top-K
        top_preds = sorted(preds, key=lambda x: -x['score'])[:top_k]
        enriched_top_preds = []
        for rel in top_preds:
            key = (tuple(rel['head_pos']), tuple(rel['tail_pos']), rel['label'])
            enriched_top_preds.append({
                'relation': enrich_relation(key),
                'score': rel['score']
            })

        # False Positives
        for rel in preds:
            key = (tuple(rel['head_pos']), tuple(rel['tail_pos']), rel['label'])
            if key not in gold_set and rel['score'] >= threshold:
                example_fp.append({
                    'relation': enrich_relation(key),
                    'score': rel['score'], 
                    'top_predictions': enriched_top_preds
                })

        # False Negatives
        for key in gold_set:
            if key not in pred_set:
                score = 0.0
                for rel in preds:
                    pred_key = (tuple(rel['head_pos']), tuple(rel['tail_pos']), rel['label'])
                    if pred_key == key:
                        score = rel['score']
                        break
                example_fn.append({
                    'relation': enrich_relation(key),
                    'score': score,
                    'top_predictions': enriched_top_preds
                })

        # Sort by score
        example_fp.sort(key=lambda x: -x['score'])
        example_fn.sort(key=lambda x: x['score'])

        error_report.append({
            'text': text,
            'gold': gold_relations,
            'false_positives': example_fp,
            'false_negatives': example_fn
        })

    return error_report

def get_top_errors(error_report, top_n=5) -> Dict:
    top_fp_examples = []
    top_fn_examples = []

    for entry in error_report:
        text = entry['text']
        gold = entry['gold']

        if entry['false_positives']:
            top_fp = entry['false_positives'][0]
            top_fp_examples.append({
                'text': text,
                'gold': gold,
                'relation': top_fp['relation'],
                'score': top_fp['score'],
                'top_predictions': top_fp['top_predictions']
            })

        if entry['false_negatives']:
            top_fn = entry['false_negatives'][0]
            top_fn_examples.append({
                'text': text,
                'gold': gold,
                'relation': top_fn['relation'],
                'score': top_fn['score'],
                'top_predictions': top_fn['top_predictions']
            })

    # Sort overall
    sorted_fp = sorted(top_fp_examples, key=lambda x: -x['score'])[:top_n]
    sorted_fn = sorted(top_fn_examples, key=lambda x: x['score'])[:top_n]

    return {
        'top_false_positives': sorted_fp,
        'top_false_negatives': sorted_fn
    }