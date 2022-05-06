from typing import Optional
import torch

from probekit.utils.types import PyTorchDevice, Specification


class Probe:
    r"""
    Defines a general (discriminative) probing model, which exposes only one value, :math:`p(\pi | h)`.
    """
    def __init__(self, device: PyTorchDevice, num_classes: int):
        self._device = device
        self._num_classes = num_classes

    def get_device(self) -> PyTorchDevice:
        """ Returns the pyTorch device this model resides in. """
        return self._device

    @classmethod
    def from_specification(cls, specification: Specification, device: PyTorchDevice = "cpu"):
        r"""Class method that is meant to be overriden to initialize the base class and the probe's
        functionality.

        Class method that is meant to be overriden to (i) initialize the base class and (ii) initialize all
        the model's parameters and functionality from a specification, which is a dictinary output by a
        Trainer instance.

        Args:
            specification (Specification): A specification containing all the information required to
                initialize the probe.
            device (PyTorchDevice): The pyTorch device that this model should be resident in.

        Returns:
            ProbingModel: The initialized probing model instance.
        """
        raise NotImplementedError

    def _log_prob_class_given_input(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Meant to be overriden by discriminative classifiers with an implementation of
        :math:`p(h \mid \pi)`.

        Args:
            inputs (torch.Tensor): An :math:`N \times d` tensor, where :math:`N` is the number of datapoints
                and :math:`d` is the dimensionality of the datapoints.

        Returns:
            log_probs (torch.Tensor): An :math:`N \times K` tensor of the log probabilities for each possible
                classification.
        """
        raise NotImplementedError

    def log_prob_class_given_input(
            self, inputs: torch.Tensor, classes: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Returns a tensor of log probabilities associated with a set of inputs for every possible class.

        Returns a tensor of classification log probabilities associated with a set of inputs for every
        possible class, :math:`p(\pi \mid h). If `classes` is provided, this returns a vector of log
        probabilities for the class provided for each input.

        Args:
            inputs (torch.Tensor): An :math:`N \times d` tensor, where :math:`N` is the number of datapoints
                and :math:`d` is the dimensionality of the datapoints.
            classes (torch.Tensor, optional): If provided, it should be an :math:`N`-dimensional vector with
                the classes for each datapoint. Defaults to None.

        Returns:
            torch.Tensor: An :math:`N \times K` tensor with the log probability of each class for each
                datapoint. If `classes` is provided, this returns a :math:`N`-dimensional vector which
                contains only the log probabilities for the specified classes.
        """
        log_probs = self._log_prob_class_given_input(inputs)

        if classes is not None:
            num_samples = inputs.shape[0]

            # Select correct sample log prob according to which value it was sampled from
            mask = torch.arange(0, self._num_classes).reshape(1, -1).expand(
                num_samples, -1).to(self._device) == classes.unsqueeze(1).expand(-1, self._num_classes)
            return torch.sum(mask * log_probs, dim=1)

        return log_probs

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Returns the predicted class ID for each input.

        Args:
            inputs (torch.Tensor): An :math:`N \times d` tensor, where :math:`N` is the number of datapoints
                and :math:`d` is the dimensionality of the datapoints.

        Returns:
            log_probs (torch.LongTensor): An :math:`N`-dimensional vector containing the ID of the predicted
                class for each datapoint.
        """
        return self.log_prob_class_given_input(inputs).argmax(dim=1)
