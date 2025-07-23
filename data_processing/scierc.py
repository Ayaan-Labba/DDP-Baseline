from nltk.tokenize.treebank import TreebankWordDetokenizer
from typing import Dict, List

# import sys
# sys.path.append("./data_processing/")

# from common import get_relation_tuples

def process_scierc(example: Dict) -> Dict:
    """ Process a single SciERC data point into GLiREL acceptable format.
    Args:
        example (Dict): A SciERC data point; should include 'tokens', 'entities', and 'relations' keys.
    Returns:
        Dict: Processed data point in GLiREL acceptable format with 'text', 'tokens', 'ner', and 'gold_relations' keys.
    """
    sentences = example['sentences']
    ex_ner = example['ner']
    ex_rel = example['relations']

    tokens = []
    ner = []
    offset = 0
    relations = []
    detokenizer = TreebankWordDetokenizer()

    for sent, ner_spans, rels in zip(sentences, ex_ner, ex_rel):
        tokens.extend(sent)
        for span in ner_spans:
            start, end, label = span
            entity_text = detokenizer.detokenize(sent[start-offset:end+1-offset])
            ner.append([start, end, label, entity_text])
        for rel in rels:
            head_start, head_end, tail_start, tail_end, label = rel
            relations.append({
            'head_pos': [head_start, head_end+1],
            'tail_pos': [tail_start, tail_end+1],
            'head_text': detokenizer.detokenize(sent[head_start-offset:head_end+1-offset]),
            'tail_text': detokenizer.detokenize(sent[tail_start-offset:tail_end+1-offset]),
            'label': label,
            'score': 1.0
        })
        offset += len(sent)

    # Reconstruct text by joining tokens
    text = detokenizer.detokenize(tokens)

    return {
        'text': text,
        'tokens': tokens,
        'ner': ner,
        'relations': relations
    }