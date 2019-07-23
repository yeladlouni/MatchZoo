"""Embedding data loader."""

from pathlib import Path

import matchzoo as mz

def load_webteb_embedding(dimension: int = 50) -> mz.embedding.Embedding:
    """
    Return the pretrained glove embedding.

    :param dimension: the size of embedding dimension, the value can only be
        50, 100, or 300.
    :return: The :class:`mz.embedding.Embedding` object.
    """
    file_name = 'fasttext.webteb.100d.vec'
    file_path = Path('matchzoo/data/embeddings').joinpath(file_name)
    return mz.embedding.load_from_file(file_path=str(file_path), mode='glove')
