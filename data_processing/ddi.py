import sys
sys.path.append("./data_processing/")

def get_ddi_labels(dataset):
    labels = set()
    for example in dataset:
        for rel in example['relations']:
            labels.add(rel['type'])
    
    return sorted(list(labels))

def create_ddi_input(example, nlp):
    passage = example['passages'][0]['text'][0]  # get the main sentence
    doc = nlp(passage)
    tokens = [token.text for token in doc]

    # Build mapping: character span → token index
    char_to_token = {}
    for i, token in enumerate(doc):
        for pos in range(token.idx, token.idx + len(token)):
            char_to_token[pos] = i

    span_to_id_map = {}  # (start_token, end_token) → id
    text_to_id_map = {}  # entity text → id

    # Convert entities from char-span to token-span
    ner = []
    for ent in example['entities']:
        char_start, char_end = ent['offsets'][0]  # char_end is exclusive
        # Get token-level span (inclusive)
        try:
            token_start = char_to_token[char_start]
            token_end = char_to_token[char_end - 1]
            ner.append([
                token_start,
                token_end,
                ent['type'],
                ent['text'][0]
            ])

            span_to_id_map[(token_start, token_end + 1)] = ent['id']
            text_to_id_map[ent['text'][0].lower().strip()] = ent['id']

        except KeyError:
            # Skip entity if char span does not map to tokens cleanly
            continue

    # Gold relations (as IDs)
    gold_relations = []
    for rel in example['relations']:
        gold_relations.append((rel['arg1_id'], rel['arg2_id'], rel['type'].lower().strip()))

    return {
        "tokens": tokens,
        "ner": ner,
        "gold_relations": gold_relations,
        "span_to_id_map": span_to_id_map,
        "text_to_id_map": text_to_id_map,
    }

def evaluate_ddi(test_data, predicted_relations, threshold=0.5):
    assert len(test_data) == len(predicted_relations)

    tp = fp = fn = 0

    for example, preds in zip(test_data, predicted_relations):
        id_map = example["span_to_id_map"]  # (start, end) → ent_id
        gold = set(example["gold_relations"])

        pred_set = set()
        for rel in preds:
            if rel["score"] < threshold:
                continue
            h_span = (rel["head_pos"][0], rel["head_pos"][1])
            t_span = (rel["tail_pos"][0], rel["tail_pos"][1])
            h_id = id_map.get(h_span)
            t_id = id_map.get(t_span)
            pred_set.add((h_id, t_id, rel["label"].lower().strip()))

        tp += len(pred_set & gold)
        fp += len(pred_set - gold)
        fn += len(gold - pred_set)

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

def evaluate_ddi_by_entity_text(test_data, predicted_relations, threshold=0.5):
    tp = fp = fn = 0

    for example, preds in zip(test_data, predicted_relations):
        id_map = example["text_to_id_map"]
        gold = set(example["gold_relations"])

        pred_rels = set()
        for rel in preds:
            if rel["score"] >= threshold:
                h_id = id_map.get(rel["head_text"][0].lower().strip())
                t_id = id_map.get(rel["tail_text"][0].lower().strip())
                label = rel["label"].lower().strip()
                pred_rels.add((h_id, t_id, label))

        tp += len(pred_rels & gold)
        fp += len(pred_rels - gold)
        fn += len(gold - pred_rels)

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

def get_ddi_errors(dataset, predictions, threshold=0.0, top_k=3):
    assert len(dataset) == len(predictions)

    error_report = []

    for example, preds in zip(dataset, predictions):
        gold_set = set(example["gold_relations"])
        tokens = example["tokens"]
        text = " ".join(tokens)
        id_map = example["span_to_id_map"]
        
        pred_set = set()
        for rel in preds:
            if rel["score"] < threshold:
                continue
            h_span = (rel["head_pos"][0], rel["head_pos"][1])
            t_span = (rel["tail_pos"][0], rel["tail_pos"][1])
            h_id = id_map.get(h_span)
            t_id = id_map.get(t_span)
            pred_set.add((h_id, t_id, rel["label"].lower().strip()))

        def enrich_relation(rel):
            h_id, t_id, label = rel
            h_span = next((span for span, eid in id_map.items() if eid == h_id), None)
            t_span = next((span for span, eid in id_map.items() if eid == t_id), None)

            h_text = tokens[h_span[0]:h_span[1]] if h_span else ["?"]
            t_text = tokens[t_span[0]:t_span[1]] if t_span else ["?"]

            return ((h_id, h_text), (t_id, t_text), label)

        # Build prediction set and top_k list
        top_preds = sorted(preds, key=lambda x: -x["score"])[:top_k]
        enriched_top_preds = []
        for rel in top_preds:
            h_span = tuple(rel["head_pos"])
            t_span = tuple(rel["tail_pos"])
            h_id = id_map.get(h_span)
            t_id = id_map.get(t_span)
            label = rel["label"].lower().strip()
            enriched_top_preds.append({
                "relation": enrich_relation((h_id, t_id, label)),
                "score": rel["score"]
            })

        # False Positives
        example_fp = []
        for rel in preds:
            if rel["score"] < threshold:
                continue
            h_span = tuple(rel["head_pos"])
            t_span = tuple(rel["tail_pos"])
            h_id = id_map.get(h_span)
            t_id = id_map.get(t_span)
            label = rel["label"].lower().strip()
            key = (h_id, t_id, label)
            if key not in gold_set:
                example_fp.append({
                    "relation": enrich_relation(key),
                    "score": rel["score"]
                })

        # False Negatives
        example_fn = []

        for gold in gold_set:
            if gold not in pred_set:
                # Try to get matching score if predicted with low confidence
                score = 0.0
                for rel in preds:
                    h_span = tuple(rel["head_pos"])
                    t_span = tuple(rel["tail_pos"])
                    h_id = id_map.get(h_span)
                    t_id = id_map.get(t_span)
                    label = rel["label"].lower().strip()
                    if (h_id, t_id, label) == gold:
                        score = rel["score"]
                        break
                example_fn.append({
                    "relation": enrich_relation(gold),
                    "score": score,
                    "top_predictions": enriched_top_preds
                })

        # Sort by score
        example_fp.sort(key=lambda x: -x["score"])
        example_fn.sort(key=lambda x: x["score"])

        if example_fp or example_fn:
            error_report.append({
                "text": text,
                "gold": [enrich_relation(r) for r in gold_set],
                "false_positives": example_fp,
                "false_negatives": example_fn
            })

    return error_report

def get_top_ddi_errors(error_report, top_n=10):
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