def get_chemprot_labels(dataset_split):
    relation_labels = set()
    for example in dataset_split:
        for rel in example['relations']:
            relation_labels.add(rel['type'])
    return sorted(list(relation_labels))

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
        "raw_text": passage,
    }