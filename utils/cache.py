from typing import Dict, List, Tuple, Optional
from utils.word import Word
import numpy as np
import torch
from trainer.base import Trainer
from tqdm import tqdm


def diagonalize_cov(cov: np.ndarray) -> np.ndarray:
    return np.diag(np.diagonal(cov))


class AttributeValueGaussianCacheEntry:
    def __init__(self, attribute: str, value: str, words: List[Word], trainer: Trainer,
                 diagonal_only: bool = False) -> None:
        self.attribute = attribute
        self.value = value
        self.trainer = trainer

        self.mean, self.cov = self.trainer.compute_gaussian_model_params_for_attribute_value(attribute, value, words)

        # This matrix may have very small eigenvalues, and due to floating point truncation errors these may be
        # negative, making the matrix non-PSD. We add small value to the diagonal elements to prevent this
        # from happening.
        try:
            min_eig = np.min(np.real(np.linalg.eigvals(self.cov)))
        except Exception as ex:
            raise Exception(ex)

        if min_eig < 0:
            self.cov -= 10 * min_eig * np.eye(self.cov.shape[0])

        if diagonal_only:
            self.cov = diagonalize_cov(self.cov)

        self.sampling_prob = self.trainer.compute_categorical_model_sampling_prob_for_attribute_value(
            attribute, value, words)
        self.sampling_prob_torch = torch.tensor([self.sampling_prob]).float()

        self.mean_torch = torch.tensor(self.mean).float()
        self.cov_torch = torch.tensor(self.cov).float()

    def get_sampling_prob(self, as_torch=True):
        if as_torch:
            return self.sampling_prob_torch
        else:
            return self.sampling_prob

    def get_gaussian_model_params(self, as_torch=True):
        if as_torch:
            return self.mean_torch, self.cov_torch
        else:
            return self.mean, self.cov

    def get_attribute(self):
        return self.attribute

    def get_value(self):
        return self.value


class AttributeValueGaussianCache:
    """
    Lots of attributes are computed based on the (attribute, value) tuple.
    This cache automated their generation.
    """
    def __init__(self, words: List[Word], trainer: Trainer, diagonal_only: bool = False,
                 attribute_values_dict: Optional[Dict[str, List[str]]] = None) -> None:
        self.words = words
        self.cache: Dict[str, Dict[str, AttributeValueGaussianCacheEntry]] = {}
        self.diagonal_only = diagonal_only
        self.trainer = trainer

        if attribute_values_dict is None:
            # Preload everything by bruteforcing every combination
            for w in tqdm(words, desc="Build Cache"):
                for a in w.get_attributes():
                    # Actual values
                    self.get_cache_entry(a, w.get_attribute(a))
        else:
            # Preload everything based on precomputed options
            for attr, vals in tqdm(attribute_values_dict.items(), desc="Build Cache"):
                for v in vals:
                    self.get_cache_entry(attr, v)

    def get_cache_entry(self, attribute: str, value: str) -> AttributeValueGaussianCacheEntry:
        if attribute not in self.cache:
            self.cache[attribute] = {}

        if value not in self.cache[attribute]:
            self.cache[attribute][value] = AttributeValueGaussianCacheEntry(
                attribute, value, self.words, trainer=self.trainer, diagonal_only=self.diagonal_only)

        return self.cache[attribute][value]

    def get_all_attribute_values(self, attribute: str) -> List[str]:
        # We want to skip dummy values
        return [k for k, v in self.cache[attribute].items()]

    def has_attribute(self, attribute: str) -> bool:
        return attribute in self.cache
