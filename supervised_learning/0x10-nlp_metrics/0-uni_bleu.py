#!/usr/bin/env python3
"""Natural Language Processing"""


import numpy as np


def uni_bleu(references, sentence):
    """uni_bleu"""
    sentence_length = len(sentence)
    references_length = []
    words = {}

    for i in references:
        references_length.append(len(i))
        for j in i:
            if j in sentence and j not in words.keys():
                words[j] = 1

    total = sum(words.values())
    index = np.argmin([abs(len(i) - sentence_length) for i in references])
    best_match = len(references[index])

    if sentence_length > best_match:
        BLEU = 1
    else:
        BLEU = np.exp(1 - float(best_match) / float(sentence_length))
    BLEU_score = BLEU * np.exp(np.log(total / sentence_length))

    return BLEU_score
