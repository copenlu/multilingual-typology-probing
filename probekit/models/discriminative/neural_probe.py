from overrides import overrides
from typing import Dict, Any, List, TYPE_CHECKING
from abc import abstractmethod

import torch
import torch.distributions
from torch import nn
import torch.nn.functional as F

from probekit.utils.types import PyTorchDevice
from probekit.models.probe import Probe


if TYPE_CHECKING:
    # Only for mypy
    TensorModule = nn.Module[torch.Tensor]
else:
    TensorModule = nn.Module


class NeuralProbeModel(TensorModule):
    r"""Interface for a neural probe model.

    Interface for a neural probe model. This is basically a standard pyTorch module,
    with some additional helper methods.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Given some input embeddings, outputs scores for each possible class.

        Args:
            input (torch.Tensor): The input embeddings, :math:`N \times D`

        Returns:
            torch.Tensor: The output scores for each class, :math:`N \times K`
        """
        raise NotImplementedError()

    def reset_parameters(self):
        r"""Re-initializes the parameters of the network.

        See: https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819
        """
        def reset_weights(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()

        self.apply(reset_weights)

    @abstractmethod
    def get_input_dim(self) -> int:
        """Returns the expected dimensionality of the input (often the embeddings).

        Returns:
            int: The dimensionality of the input
        """
        raise NotImplementedError()

    @abstractmethod
    def get_output_dim(self) -> int:
        """Returns the expected dimensionality of the output (often the number of classes/property values).

        Returns:
            int: The dimensionality of the output
        """
        raise NotImplementedError()


class MLPProbeModel(NeuralProbeModel):
    def __init__(
            self, embedding_size: int, num_classes: int, hidden_size: int,
            num_layers: int, activation=nn.Sigmoid):
        super().__init__()

        self._embedding_size = embedding_size
        self._num_classes = num_classes

        if num_layers < 1:
            raise Exception("Need at least one layer in MLP.")

        # Build list of layer dimensionalities
        dimensionalities: List[int] = []
        dimensionalities += [embedding_size]
        dimensionalities += [hidden_size] * (num_layers - 1)
        dimensionalities += [num_classes]

        # Build modules
        layers: List[nn.Module] = [nn.Linear(dimensionalities[0], dimensionalities[1])]
        for dim_1, dim_2 in zip(dimensionalities[1:], dimensionalities[2:]):
            layers += [activation(), nn.Linear(dim_1, dim_2)]

        self._layers = nn.Sequential(*layers)

    @overrides
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._layers(input)

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_size

    @overrides
    def get_output_dim(self) -> int:
        return self._num_classes


class MaskAndMapWrapper(NeuralProbeModel):
    def __init__(self, model: NeuralProbeModel, select_dimensions: List[int]):
        super().__init__()

        self._model = model
        self._device = next(model.parameters()).device
        self._select_dimensions = select_dimensions
        self._indices_tensor = torch.tensor(self._select_dimensions).reshape(1, -1).to(self._device)

    @overrides
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        indices_tensor = self._indices_tensor.expand(batch_size, -1)
        scattered_inputs = torch.zeros(batch_size, self._model.get_input_dim()).to(self._device)
        scattered_inputs = scattered_inputs.scatter(dim=1, index=indices_tensor, src=inputs)
        return self._model(scattered_inputs)

    @overrides
    def get_input_dim(self) -> int:
        return len(self._select_dimensions)

    @overrides
    def get_output_dim(self) -> int:
        return self._model.get_output_dim()


class NeuralProbe(Probe):
    def __init__(self, model: NeuralProbeModel, device: PyTorchDevice = "cpu"):
        self._model = model.to(device)

        super().__init__(device=device, num_classes=self._model.get_output_dim())

    @classmethod
    def from_specification(cls, parameters: Dict[str, Any], device: PyTorchDevice = "cpu"):
        return cls(model=parameters["model"], device=device)

    def get_underlying_model(self) -> NeuralProbeModel:
        return self._model

    @overrides
    def _log_prob_class_given_input(self, inputs: torch.Tensor):
        """Computes the class probabilities according to an arbitrary neural probe.

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
        assert inputs.shape[1] == self._model.get_input_dim()

        # shape: (num_samples, num_classes)
        class_scores = self._model.forward(inputs)

        assert class_scores.shape[0] == num_samples
        assert class_scores.shape[1] == self._num_classes

        # shape: (num_samples, num_classes)
        log_probs = F.log_softmax(class_scores, dim=1)
        assert log_probs.shape == (num_samples, self._num_classes)

        return log_probs
