from typing import Optional, Dict
from overrides import overrides
import numpy as np
import pickle
from os import path


class Embedder:
    def __init__(self, cache_file: Optional[str] = None, update_cache: bool = False) -> None:
        self._cache_file = cache_file or None
        self._update_cache = update_cache

        # Load cache if it is enabled.
        # Create empty cache if it doesn't exist (though this doesn't create a file)
        self._cache: Dict[str, np.ndarray] = {}
        if self._cache_file is not None and path.exists(self._cache_file):
            with open(self._cache_file, "rb") as h:
                self._cache = pickle.load(h)

    def update_cache_file(self) -> None:
        if self._cache_file is None:
            raise Exception("Cannot update cache as it has not been enabled.")

        with open(self._cache_file, "wb") as h:
            pickle.dump(self._cache, h)

    def get_embedding(self, word: str) -> np.ndarray:
        if self.has_cached_word(word):
            return self._cache[word]

        # Write new word to cache
        embedding = self.compute_embedding(word)
        self._cache[word] = embedding

        # Update cache
        if self._update_cache:
            self.update_cache_file()

        return embedding

    def compute_embedding(self, word: str) -> np.ndarray:
        raise NotImplementedError

    def get_dimensionality(self) -> int:
        raise NotImplementedError

    def has_cached_word(self, word: str) -> bool:
        if word in self._cache:
            return True

        return False