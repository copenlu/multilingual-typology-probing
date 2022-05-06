from typing import Dict
import numpy as np


class Word:
    def __init__(self, word: str, embedding: np.ndarray, count: int, attributes: Dict[str, str]) -> None:
        self._word = word
        self._embedding = embedding
        self._count = count
        self._attributes = attributes

    def get_word(self) -> str:
        return self._word

    def get_embedding(self):
        return self._embedding

    def get_count(self):
        return self._count

    def has_attribute(self, attr):
        return attr in self._attributes

    def get_attribute(self, attr):
        return self._attributes[attr]

    def get_attributes(self):
        return list(self._attributes.keys())

    def __repr__(self) -> str:
        return "{}({})".format(self._word, self._attributes)
