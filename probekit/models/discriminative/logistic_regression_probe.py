from overrides import overrides
from typing import Dict, Any

import torch
import torch.distributions
import torch.nn.functional as F

from probekit.utils.types import PyTorchDevice
from probekit.models.probe import Probe


class LogisticRegressionProbe(Probe):
    def __init__(self, weights: torch.Tensor, biases: torch.Tensor,
                 device: PyTorchDevice = "cpu"):
        self._embedding_size = weights.shape[0]
        self._weights = weights.to(device)
        self._biases = biases.to(device)

        super().__init__(device=device, num_classes=biases.shape[0])

    @classmethod
    def from_specification(
            cls, parameters: Dict[str, Any],
            device: PyTorchDevice = "cpu"):

        return cls(weights=parameters["weights"], biases=parameters["biases"], device=device)

    @overrides
    def _log_prob_class_given_input(self, inputs: torch.Tensor):
        """Computes the class probabilities according to a logistic regression classifier.

        Args:
            inputs (torch.Tensor): An :math:`N \times d` tensor, where :math:`N` is the number of datapoints
                and :math:`d` is the dimensionality of the datapoints.

        Returns:
            torch.Tensor: An :math:`N \times K` tensor where :math:`K` is the number of classes. Each entry
                contains the log probability of a datapoint being assigned as specific class.
        """
        num_samples = inputs.shape[0]

        # inputs shape: (num_samples, dimensionality)
        assert inputs.shape[0] == num_samples
        assert inputs.shape[1] == self._embedding_size

        # shape: (num_samples, num_classes)
        class_scores = inputs @ self._weights + self._biases
        assert class_scores.shape[0] == num_samples
        assert class_scores.shape[1] == self._num_classes

        # shape: (num_samples, num_classes)
        log_probs = F.log_softmax(class_scores, dim=1)
        assert log_probs.shape == (num_samples, self._num_classes)

        return log_probs
