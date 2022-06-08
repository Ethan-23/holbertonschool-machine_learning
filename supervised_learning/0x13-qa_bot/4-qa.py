#!/usr/bin/env python3
"""Semantic Search QA"""

import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from transformers import BertTokenizer


def question_answer(corpus_path):
    """Finds best answer for given question"""
    exit_list = ["exit", "quit", "goodbye", "bye"]

    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    while True:
        question = input("Q: ")
        if question.lower() in exit_list:
            break
        reference_tokens = tokenizer.tokenize(
            semantic_search(corpus_path, question))
        question_tokens = tokenizer.tokenize(question)
        tokens = ['[CLS]'] + question_tokens + ['[SEP]']\
            + reference_tokens + ['[SEP]']

        input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_word_ids)
        input_type_ids = [0] * (1 + len(question_tokens) + 1)\
            + [1] * (len(reference_tokens) + 1)

        input_word_ids, input_mask, input_type_ids =\
            map(lambda t: tf.expand_dims(
                tf.convert_to_tensor(t, dtype=tf.int32), 0),
                (input_word_ids, input_mask, input_type_ids))

        outputs = model([input_word_ids, input_mask, input_type_ids])
        short_start = tf.argmax(outputs[0][0][1:]) + 1
        short_end = tf.argmax(outputs[1][0][1:]) + 1
        answer_tokens = tokens[short_start: short_end + 1]
        if(answer_tokens == [] or answer_tokens[-1] == "[SEP]"):
            print("A: Sorry, I do not understand your question.")
            continue

        answer = tokenizer.convert_tokens_to_string(answer_tokens)
        print("A: {}".format(answer))
    print("A: Goodbye")


def semantic_search(corpus_path, sentence):
    """semantic_search"""
    documents = [sentence]
    for filename in os.listdir(corpus_path):
        if filename.endswith(".md") is False:
            continue
        with open(corpus_path + "/" + filename, "r", encoding="utf-8") as f:
            documents.append(f.read())

    model = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5")

    embeddings = model(documents)
    correlation = np.inner(embeddings, embeddings)
    closest = np.argmax(correlation[0, 1:])
    similar = documents[closest + 1]

    return similar
