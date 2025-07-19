import json
from glirel import GLiREL

def get_pred_set(prediction, threshold=0.5):
    pred = set()
    for rel in prediction:
        if rel["score"] < threshold:
            continue
        pred.add((tuple(rel["head_pos"]), tuple(rel["tail_pos"]), rel["label"]))
    
    return pred

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]
    
def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def run_inference(model, input, labels, threshold=0.5, top_k=1):
    print("Running inference on device:", model.device)

    predictions = []
    for example in input:
        tokens = example["tokens"]
        ner = example["ner"]
        prediction = model.predict_relations(tokens, labels, threshold=threshold, ner=ner, top_k=top_k)
        predictions.append(prediction)
    
    return predictions