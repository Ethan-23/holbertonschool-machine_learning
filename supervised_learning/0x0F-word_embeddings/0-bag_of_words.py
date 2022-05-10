#!/usr/bin/env python3
"""0x0F. Natural Language Processing - Word Embeddings"""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """bag_of_words"""
    countVector = CountVectorizer(vocabulary=vocab)
    fitTransform = countVector.fit_transform(sentences)
    embeddings = fitTransform.toarray()
    features = countVector.get_feature_names()
    return embeddings, features
