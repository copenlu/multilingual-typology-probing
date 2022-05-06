from typing import Optional, List

from probekit.utils.types import PyTorchDevice, Specification, PropertyValue
from probekit.utils.dataset import ClassificationDataset


class Trainer:
    def __init__(self, dataset: ClassificationDataset, device: PyTorchDevice):
        self._dataset = dataset
        self._property_values = list(dataset.keys())
        self._device = device
        self._total_dimensions = self._dataset.get_dimensionality()

    def get_device(self) -> PyTorchDevice:
        return self._device

    def get_property_values(self) -> List[PropertyValue]:
        return self._property_values

    def train(self):
        self._train()

    def _train(self):
        raise NotImplementedError

    def get_specification(self, select_dimensions: Optional[List[int]] = None) -> Specification:
        dims = select_dimensions
        if dims is None:
            dims = list(range(self._total_dimensions))

        specification = self._get_specification(dims)
        specification["_selected_dimensions"] = select_dimensions
        return specification

    def _get_specification(self, select_dimensions: List[int]) -> Specification:
        raise NotImplementedError
