from overrides import overrides
from typing import List, Optional

import torch
import torch.distributions
from torch import nn

from probekit.models.discriminative.neural_probe import NeuralProbeModel


class MaskableNeuralProbeModel(NeuralProbeModel):
    r"""Interface for a maskable neural probe model.

    Interface for a maskable neural probe model. This is basically a standard pyTorch module,
    with some additional helper methods. The main difference between this an NeuralProbeModel
    is that this allows an optional mask to be supplied.

    This allows for behaviour that is not possible with the MaskAndMapWrapper.
    """
    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Given some input embeddings, outputs scores for each possible class.

        Args:
            input (torch.Tensor): The input embeddings, :math:`N \times D`
            mask (Optional[torch.Tensor]): The mask to apply, :math:`N \times D` (or broadcastable to it).

        Returns:
            torch.Tensor: The output scores for each class, :math:`N \times K`
        """
        raise NotImplementedError()


class DecomposableLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)

        raise NotImplementedError("This layer doesn't work well. Usage not recommended.")

        if bias:
            # Need to override bias so that it is decomposable
            self.bias = nn.Parameter(torch.Tensor(in_features))
            self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()

        # Rescale magnitude of bias dimensions
        #if self.bias is not None:
        #    dim = self.bias.shape[0]
        #    self.bias.data /= dim

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(input)

        if self.bias is None:
            return super().forward(input * mask)

        assert input.shape == mask.shape
        batch_size, dim = input.shape

        # shape: (batch_size, in_features, out_features)
        expanded_weight = self.weight.T
        assert expanded_weight.shape == (self.in_features, self.out_features)

        return ((input - self.bias) * mask) @ expanded_weight 

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, decomposable'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class MLPProbeModel(MaskableNeuralProbeModel):
    def __init__(
            self, embedding_size: int, num_classes: int, hidden_size: int,
            num_layers: int, activation: str, decomposable: bool = True):
        super().__init__()

        self._embedding_size = embedding_size
        self._num_classes = num_classes
        self._decomposable = decomposable

        if num_layers < 1:
            raise Exception("Need at least one layer in MLP.")
        
        # Build list of layer dimensionalities
        dimensionalities: List[int] = []
        dimensionalities += [embedding_size]
        dimensionalities += [hidden_size] * (num_layers - 1)
        dimensionalities += [num_classes]

        # Build modules
        if decomposable:
            self._first_layer: nn.Module = DecomposableLinear(dimensionalities[0], dimensionalities[1])
        else:
            self._first_layer: nn.Module = nn.Linear(dimensionalities[0], dimensionalities[1])

        activation = nn.ReLU if activation == "relu" else nn.Sigmoid
        layers: List[nn.Module] = []
        for dim_1, dim_2 in zip(dimensionalities[1:], dimensionalities[2:]):
            layers += [activation(), nn.Linear(dim_1, dim_2)]

        self._final_layers = nn.Sequential(*layers)

    @overrides
    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self._decomposable:
            input = self._first_layer(input, mask)
        else:
            input = input * mask if mask is not None else input  # noqa
            input = self._first_layer(input)

        return self._final_layers(input)

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_size

    @overrides
    def get_output_dim(self) -> int:
        return self._num_classes
