#!/usr/bin/env python3
"""Dataset"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Dataset Class"""

    def __init__(self):
        """init for dataset"""
        self.data_train = tfds.load("ted_hrlr_translate/pt_to_en",
                                    split="train",
                                    as_supervised=True)
        self.data_valid = tfds.load("ted_hrlr_translate/pt_to_en",
                                    split="validation",
                                    as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """tokenize_dataset"""
        SubwordTextEncoder = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_en = SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data),
            target_vocab_size=(2 ** 15))
        tokenizer_pt = SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data),
            target_vocab_size=(2 ** 15))
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """encode"""
        pt_index = self.tokenizer_pt.vocab_size
        en_index = self.tokenizer_en.vocab_size
        pt_tokens = [pt_index] + self.tokenizer_pt.encode(
            pt.numpy()) + [pt_index + 1]
        en_tokens = [en_index] + self.tokenizer_en.encode(
            en.numpy()) + [en_index + 1]
        return pt_tokens, en_tokens
