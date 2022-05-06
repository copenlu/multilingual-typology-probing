from typing import List, Optional, Dict, Tuple, Mapping, Sequence, TypeVar
from overrides import overrides
from abc import abstractmethod
import torch
import random
import copy
import warnings
from collections import Counter
from itertools import groupby
import operator

from probekit.utils.types import PyTorchDevice, PropertyValue, Word, Arc, Inference


T = TypeVar("T", Word, Arc, Inference)
ClassificationDatasetDatastore = Mapping[PropertyValue, Sequence[T]]


class DatasetTransform:
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def __call__(self, input: torch.Tensor):
        return self.transform(input)

    @abstractmethod
    def to(self, device: PyTorchDevice):
        raise NotImplementedError()


class ClassificationDataset(Mapping[PropertyValue, Sequence[T]]):
    """
    A classification dataset amounts to an immutable dictionary, alongside some helper methods.

    See: https://stackoverflow.com/questions/21361106/how-would-i-implement-a-dict-with-abstract-base-classes-in-python  # noqa

    The `transform` is an optional argument that can be used to apply a transformation to every datapoint.
    """
    def __init__(self, data: ClassificationDatasetDatastore[T], device: PyTorchDevice = "cpu",
                 transform: Optional[DatasetTransform] = None):
        self._data: ClassificationDatasetDatastore[T] = copy.deepcopy(data)
        self._device = device
        self._transform = transform.to(device) if transform else None

        for property_value, word_list in self._data.items():
            if len(word_list) <= 0:
                raise Exception("Datastore cannot be sparse. Each possible property value must have at least"
                                " one training example.")

        # Generate tensorized version of the each properties embeddings
        self._embeddings_tensors = {
            property_value: torch.stack([w.get_embedding() for w in word_list], dim=0).to(self._device)
            for property_value, word_list in self._data.items()
        }

        # Generate concatenated versions of the inputs, values and words
        self._embeddings_tensor_concat = torch.cat(
            [tensor for _, tensor in self._embeddings_tensors.items()], dim=0).to(self._device)
        self._values_tensor_concat = torch.cat(
            [torch.tensor([idx] * len(words))
             for idx, (_, words) in enumerate(self._data.items())], dim=0).to(self._device)

        if transform is not None:
            self._embeddings_tensor_concat = self._transform(self._embeddings_tensor_concat)

            # Update per-property tensor
            self._embeddings_tensors = {prop: self._transform(tensor) for prop, tensor in
                                        self._embeddings_tensors.items()}

    @staticmethod
    def get_property_value_list(attribute: str, *word_lists, min_count: int = 1) -> List[PropertyValue]:
        property_value_counters = [
            Counter([w.get_attribute(attribute) for w in word_list if w.has_attribute(attribute)])
            for word_list in word_lists]

        # 1. Build list of property values in all datasets
        property_value_sets = [set(pcv.keys()) for pcv in property_value_counters]
        kept_property_values = list(property_value_sets[0].intersection(*property_value_sets[1:]))

        # 2. Filter by counts
        final_property_values = []
        if min_count is not None:
            for kpv in kept_property_values:
                if all([pvc[kpv] >= min_count for pvc in property_value_counters]):
                    final_property_values.append(kpv)
        else:
            final_property_values = kept_property_values

        return sorted(final_property_values)

    @classmethod
    def from_dataset(cls, dataset, device: PyTorchDevice = "cpu"):
        return cls(data=dataset._data, device=device, transform=dataset._transform)

    @classmethod
    def from_word_list(cls, *args, **kwargs):
        raise Exception("This has been deprecated. Use `from_unit_list` instead.")

    @classmethod
    def from_unit_list(cls, units: Sequence[T], attribute: str, device: PyTorchDevice = "cpu",
                       transform: Optional[DatasetTransform] = None,
                       property_value_list: Optional[List[PropertyValue]] = None):

        if property_value_list is None:
            warnings.warn("You have not specified a `property_value_list` to construct the "
                          "ClassificationDataset. This can lead to problems!")

        datastore: Dict[PropertyValue, List[T]] = {}
        if property_value_list is not None:
            datastore = {prop: [] for prop in property_value_list}

        for u in units:
            if not u.has_attribute(attribute):
                continue

            property_value: PropertyValue = u.get_attribute(attribute)
            if property_value_list is None and property_value not in datastore:
                # We are constructing the dictionary completely from scratch, so add this key
                datastore[property_value] = []

            if property_value in datastore:
                datastore[property_value].append(u)

        # Ensure keys are in alphabetical order
        datastore = dict(sorted(datastore.items()))

        return cls(datastore, device=device, transform=transform)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        internal_string = ', '.join([f"{prop} [{len(words)}]" for prop, words in self._data.items()])
        return f"{type(self).__name__}({internal_string})"

    def get_dimensionality(self) -> int:
        return self._embeddings_tensor_concat.shape[1]

    def get_device(self) -> PyTorchDevice:
        return self._device

    def get_embeddings_tensors(
            self, select_dimensions: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        if select_dimensions:
            return {prop: tensor[:, select_dimensions] for prop, tensor in self._embeddings_tensors.items()}

        return self._embeddings_tensors

    def get_inputs_values_tensor(
            self, select_dimensions: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._embeddings_tensor_concat
        if select_dimensions:
            embeddings = self._embeddings_tensor_concat[:, select_dimensions]

        return embeddings, self._values_tensor_concat


class WordClassificationDataset(ClassificationDataset):
    def __init__(self, data: ClassificationDatasetDatastore[Word], device: PyTorchDevice = "cpu",
                 transform: Optional[DatasetTransform] = None):
        super().__init__(data=data, device=device, transform=transform)

        self._data: ClassificationDatasetDatastore[Word]

        # Generate list of words. Needed for, e.g., uniform sampling.
        self._words_lists = {
            property_value: [w.get_word() for w in word_list]
            for property_value, word_list in self._data.items()
        }

        self._words_list_concat = [i for _, wl in self._words_lists.items() for i in wl]

    def get_words(self) -> Sequence[str]:
        return self._words_list_concat


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).

    Credits to: Jesse Mu (jayelm).
    See: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/5
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.device = tensors[0].device
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.dataset_len = self.tensors[0].shape[0]

    def subsample_inputs(self) -> torch.Tensor:
        """
        Subsamples a set of training point indices to be included during training. This is overloaded in,
        e.g., the UniformFastTensorDataLoader
        """
        return torch.arange(self.dataset_len).to(self.device)

    def get_effective_iteration_stats(self, active_indices) -> Dict[str, int]:
        """
        Computes the number of batches in this iteration over the training set.
        """
        effective_length = active_indices.shape[0]
        n_batches, remainder = divmod(effective_length, self.batch_size)
        if remainder > 0:
            n_batches += 1

        return {
            # Number of batches in this iteration over the dataset
            "n_batches": n_batches,

            # Effective length of this iteration over the dataset (accounting active indices)
            "effective_length": effective_length,
        }

    def __iter__(self):
        self.active_indices = self.subsample_inputs()
        iteration_stats = self.get_effective_iteration_stats(self.active_indices)
        self.n_batches, self.effective_len = iteration_stats["n_batches"], iteration_stats["effective_length"]

        if self.shuffle:
            shuffled_active_indices = torch.randperm(len(self.active_indices), device=self.device)
            self.indices = self.active_indices[shuffled_active_indices].to(self.device)
        else:
            self.indices = self.active_indices

        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.effective_len:
            raise StopIteration

        indices = self.indices[self.i:self.i + self.batch_size]
        batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.get_effective_iteration_stats(self.subsample_inputs())["n_batches"]


class FastUniformTensorDataLoader(FastTensorDataLoader):
    """
    Extension of the FastTensorDataLoader which additionally ensures that types are sampled uniformly.
    """
    def __init__(self, words, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastUniformTensorDataLoader.

        :param words: words corresponding to each datapoint in the tensors
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :returns: A FastUniformTensorDataLoader.
        """
        super().__init__(*tensors, batch_size=batch_size, shuffle=shuffle)

        # Generate mapping from type to indices of types in the dataset.
        word_fn = operator.itemgetter(1)
        self.type_to_indices_mapping = {
            k: list([x[0] for x in lst]) for k, lst in groupby(sorted(enumerate(words), key=word_fn), word_fn)
        }

    @overrides
    def subsample_inputs(self) -> torch.Tensor:
        """
        Subsamples one example for each type.
        """
        return torch.tensor(
            [random.sample(v, 1)[0] for k, v in self.type_to_indices_mapping.items()]).to(self.device)
