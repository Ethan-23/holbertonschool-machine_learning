#!/usr/bin/env python3
"""0x0F. Natural Language Processing - Word Embeddings"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """tf_idf"""
    tfidfVectorizer = TfidfVectorizer(vocabulary=vocab)
    fitTransform = tfidfVectorizer.fit_transform(sentences)
    embeddings = fitTransform.toarray()
    features = tfidfVectorizer.get_feature_names()
    return embeddings, features
