import sys
sys.path.append("./data_processing/")

def get_chemprot_labels(dataset):
    labels = set()
    for example in dataset:
        for rel in example['relations']:
            labels.add(rel['type'])
    
    return sorted(list(labels))

def create_chemprot_input(example, nlp):
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

def evaluate_chemprot(test_data, predicted_relations, threshold=0.5):
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

def evaluate_chemprot_by_entity_text(test_data, predicted_relations, threshold=0.5):
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