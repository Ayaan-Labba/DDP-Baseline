from nltk.tokenize.treebank import TreebankWordDetokenizer
from typing import Dict

# import sys
# sys.path.append("./data_processing/")

# from common import get_relation_tuples

def process_nyt(example: Dict) -> Dict:
    """ Process a single NYT data point into GLiREL acceptable format.
    Args:
        example (Dict): A NYT data point; should include 'tokens', 'entities', and 'relations' keys.
    Returns:
        Dict: Processed data point in GLiREL acceptable format with 'text', 'tokens', 'ner', and 'gold_relations' keys.
    """
    seen_spans = set()
    unique_ner = []
    detokenizer = TreebankWordDetokenizer()
    tokens = example['tokenized_text']

    for span in example['ner']:
        start = span[0]
        end = span[1] - 1 # End inclusive
        label = span[2]
        text = span[3]
        
        if (start, end) not in seen_spans:
            seen_spans.add((start, end))
            unique_ner.append([start, end, label, text])

    # Convert relations to GLiREL output format (exclusive end indices)
    relations = []
    for rel in example['relations']:
        head_ent = rel['head']
        tail_ent = rel['tail']
        relations.append({
            'head_pos': head_ent['position'],
            'tail_pos': tail_ent['position'],
            'head_text': head_ent['mention'],
            'tail_text': tail_ent['mention'],
            'label': rel['relation_text'],
            'score': 1.0
        })

    return {
        'text': detokenizer.detokenize(example['tokenized_text']),
        'tokens': tokens,
        'ner': unique_ner,
        'relations': relations
    }