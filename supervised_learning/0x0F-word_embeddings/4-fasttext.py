#!/usr/bin/env python3
"""0x0F. Natural Language Processing - Word Embeddings"""

from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5,
                   window=5, cbow=True, iterations=5, seed=0, workers=1):
    """gensim_to_keras"""
    model = FastText(sentences=sentences, size=size, min_count=min_count,
                     workers=workers, negative=negative, cbow_mean=cbow,
                     window=window, seed=seed, iter=iterations)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)
    return model
