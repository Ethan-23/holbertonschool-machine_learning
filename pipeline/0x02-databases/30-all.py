#!/usr/bin/env python3
"""List all documents in Python"""


def list_all(mongo_collection):
    """list_all"""
    docs = []
    collection = mongo_collection.find()
    for doc in collection:
        docs.append(doc)
    return docs
