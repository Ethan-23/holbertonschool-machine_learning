#!/usr/bin/env python3
"""0x0F. Natural Language Processing - Word Embeddings"""


def gensim_to_keras(model):
    """gensim_to_keras"""
    return model.wv.get_keras_embedding(train_embeddings=True)
