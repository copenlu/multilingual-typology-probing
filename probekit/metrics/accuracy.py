from typing import Optional, List
from overrides import overrides

from probekit.utils.dataset import ClassificationDataset
from probekit.metrics.metric import Metric
from probekit.models.probe import Probe


class Accuracy(Metric):
    def __init__(self, dataset: ClassificationDataset):
        super().__init__()

        self._dataset = dataset

    @overrides
    def _compute(self, probe: Probe, select_dimensions: Optional[List[int]] = None, 
                mask_class: Optional[List[int]] = None) -> float:
        inputs, true = self._dataset.get_inputs_values_tensor(select_dimensions)
        num_samples = inputs.shape[0]
        predicted = probe.predict(inputs)
        return (predicted == true).float().sum().item() / num_samples
