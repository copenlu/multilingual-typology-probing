from typing import Optional, List, Dict, Any, Union
from overrides import overrides
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
import numpy as np
import warnings

from probekit.utils.types import PyTorchDevice, Specification
from probekit.utils.dataset import FastTensorDataLoader, ClassificationDataset
from probekit.trainers.neural_probe_trainer import NeuralProbeTrainer
from probekit.models.discriminative.neural_probe import NeuralProbeModel


class FlexibleNeuralProbeTrainer(NeuralProbeTrainer):
    def __init__(
            self, model: NeuralProbeModel, dataset: ClassificationDataset, device: PyTorchDevice,
            decomposable: bool = True, lr: float = 1e-2, num_epochs: int = 2000,
            batch_size: Optional[int] = None, report_progress: bool = True, temperature: float = 1.0,
            temp_annealing: bool = True, temp_min: float = 0.5, anneal_rate: float = 0.00003,
            patience: int = 50, l1_weight: float = 0.0, l2_weight: float = 0.0,
            additional_parameters: List[nn.Parameter] = [], es_held_out_size: Optional[float] = 0.1):

        self._temperature = temperature
        self._temp_annealing = temp_annealing
        self._temp_min = temp_min
        self._anneal_rate = anneal_rate
        self._patience = patience
        self._additional_parameters = additional_parameters
        self._l1_weight = l1_weight
        self._l2_weight = l2_weight
        self._es_held_out_size = es_held_out_size

        super().__init__(
            model=model, dataset=dataset, device=device, decomposable=decomposable, lr=lr,
            num_epochs=num_epochs, batch_size=batch_size)

    def _sample_mask(self, shape, temperature: float = 1.0, device: PyTorchDevice = "cpu") -> torch.Tensor:
        raise NotImplementedError()

    def _get_regularization(self) -> torch.Tensor:
        return torch.tensor(0.0).to(self.get_device())

    def _reset_mask_parameters(self):
        pass

    def setup_parameter_list(self):
        params_to_optimize = list(self._model.parameters())
        params_to_optimize.extend(self._additional_parameters)
        return params_to_optimize

    @overrides
    def _train_for_dimensions(self, select_dimensions: Optional[List[int]] = None):
        inputs_tensor, values_tensor = self._dataset.get_inputs_values_tensor()
        num_samples, dim = inputs_tensor.shape

        model = self._model
        model.reset_parameters()
        self._reset_mask_parameters()

        params_to_optimize = self.setup_parameter_list()
        optimizer = optim.Adam(params_to_optimize, lr=self._lr)

        num_train = inputs_tensor.shape[0]
        if self._es_held_out_size is not None:
            num_train = int((1 - self._es_held_out_size) * inputs_tensor.shape[0])

        # Shuffle tensor beforehand
        shuffled_indices = torch.randperm(inputs_tensor.shape[0])

        inputs_tensor_shuffled = inputs_tensor[shuffled_indices]
        values_tensor_shuffled = values_tensor[shuffled_indices]

        data_loader = FastTensorDataLoader(
            inputs_tensor_shuffled[:num_train], values_tensor_shuffled[:num_train],
            batch_size=self._batch_size, shuffle=True)
        data_loader_es = FastTensorDataLoader(
            inputs_tensor_shuffled[num_train:], values_tensor_shuffled[num_train:],
            batch_size=self._batch_size, shuffle=True)

        print("Total dataset size:", inputs_tensor_shuffled.shape[0])
        print("Actual train dataset size:", inputs_tensor_shuffled[:num_train].shape[0])
        print("Early stopping dataset size:", inputs_tensor_shuffled[num_train:].shape[0])

        t = trange(self._num_epochs, desc="Training flexible neural probe", disable=not self._report_progress,
                   leave=False)
        # Early stopping parameters
        min_loss = np.Inf
        epochs_no_improve = 0
        temperature = self._temperature

        for epoch in t:
            es_loss = 0.0

            # Training loop
            for minibatch_idx, (minibatch, minibatch_assignment) in enumerate(data_loader):
                # loss_bp is used for backpropagation
                # loss_true is reported
                loss_bp, loss_true = self.train_loss(
                    minibatch, minibatch_assignment, select_dimensions, {"temperature": temperature})
                if self._es_held_out_size is None:
                    es_loss += loss_true
                    warnings.warn("Doing early stopping without held out data.")

                optimizer.zero_grad()
                loss_bp.backward()
                optimizer.step()

            # Validation loop
            if self._es_held_out_size is not None:
                with torch.no_grad():
                    for minibatch_idx, (minibatch, minibatch_assignment) in enumerate(data_loader_es):
                        loss_bp, loss_true = self.validation_loss(
                            minibatch, minibatch_assignment, select_dimensions, {"temperature": temperature})
                        es_loss += loss_true

            # Anneal temperature
            if self._temp_annealing:
                temperature = max(temperature - self._anneal_rate, self._temp_min)

            # EARLY STOPPING
            # Compute average held-out loss
            es_data_len = len(data_loader_es) if self._es_held_out_size is not None else len(data_loader)
            es_loss = es_loss / es_data_len

            # If loss is at a minimum
            if es_loss < min_loss:

                # Save the model
                torch.save(model, "current_model.pkl")
                epochs_no_improve = 0
                min_loss = es_loss
            else:
                epochs_no_improve += 1

                # Check early stopping condition
                if epochs_no_improve == self._patience:
                    print('Early stopping!')

                    # Load in the best model
                    model = torch.load("current_model.pkl")

                    break

            t.set_postfix(loss_true=es_loss.item(), epochs_no_improve=epochs_no_improve,
                          temperature=temperature)

    def train_loss(self, minibatch: torch.Tensor, minibatch_assignment: torch.Tensor,
                   select_dimensions: Optional[List[int]], others: Dict[str, Any]):
        # If select_dimensions is set, we want to train the probe statically on a specific set of dimensions
        # (as it isn't decomposable).
        minibatch_size, dim = minibatch.shape
        num_values = len(self.get_property_values())
        model = self._model
        temperature = others["temperature"]
        loss_fn = torch.nn.CrossEntropyLoss()

        if select_dimensions:
            mask = torch.zeros(dim).scatter(0, torch.tensor(select_dimensions), 1.0).to(self.get_device())

        # If we haven't specified any dimensions to train on, then we want to sample a mask
        if not select_dimensions:
            mask = self._sample_mask(shape=minibatch.shape, device=self._device,
                                     temperature=temperature)

        class_scores = model(minibatch, mask)
        assert class_scores.shape == (minibatch_size, num_values)

        l1_weight, l2_weight = self._l1_weight, self._l2_weight
        weights_regularization: Union[torch.Tensor, float] = 0.0
        for p in model.parameters():
            weights_regularization += l1_weight * p.abs().sum() + l2_weight * (p ** 2).sum()

        mask_regularization = self._get_regularization()
        penalty = weights_regularization + mask_regularization
        loss = loss_fn(class_scores, minibatch_assignment) + penalty

        return loss, loss

    def validation_loss(self, minibatch: torch.Tensor, minibatch_assignment: torch.Tensor,
                        select_dimensions: Optional[List[int]], others: Dict[str, Any]):
        loss_bp, loss_true = self.train_loss(minibatch, minibatch_assignment, select_dimensions, others)
        return loss_bp, loss_true

    def _get_specification(self, select_dimensions: List[int]) -> Specification:
        # If we have not forced the neural probe to be decomposable, we must train it from scratch
        # for every new selection of dimensions
        if not self._decomposable:
            self._train_for_dimensions(select_dimensions)

        # Apply a masking and mapping layer to the model's inputs.
        masked_model = MaskAndMapWrapper(self._model, select_dimensions, pass_mask=True)

        return {
            "model": masked_model
        }


class MaskAndMapWrapper(NeuralProbeModel):
    """
    This wraps a model so that it accepts inputs only from a specific subset of dimensions in the form of
    a list `select_dimensions`. This layer will ensure that whenever this model is called on a set of inputs
    (e.g., a 3-d tensor containing only the dimensions [5, 10, 3]), all but those dimensions are set to zero,
    and the dimensions provided by the tensor are mapped to the expected locations.

    Moreover, is `pass_mask` is set, this also passes the appropriate mask to the model. Some layers,
    e.g. the decomposable linear layer, expect this.
    """
    def __init__(self, model: NeuralProbeModel, select_dimensions: List[int], pass_mask: bool = False):
        super().__init__()

        self._model = model
        self._device = next(model.parameters()).device
        self._select_dimensions = select_dimensions
        self._indices_tensor = torch.tensor(self._select_dimensions).reshape(1, -1).to(self._device)
        self._pass_mask = pass_mask

    @overrides
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        indices_tensor = self._indices_tensor.expand(batch_size, -1)
        scattered_inputs = torch.zeros(batch_size, self._model.get_input_dim()).to(self._device)
        scattered_inputs = scattered_inputs.scatter(dim=1, index=indices_tensor, src=inputs)

        if self._pass_mask:
            scattered_mask = torch.zeros(batch_size, self._model.get_input_dim()).to(self._device)
            ones = torch.ones_like(inputs)
            scattered_mask = scattered_mask.scatter(dim=1, index=indices_tensor, src=ones)
            return self._model(scattered_inputs, scattered_mask)

        return self._model(scattered_inputs)

    @overrides
    def get_input_dim(self) -> int:
        return len(self._select_dimensions)

    @overrides
    def get_output_dim(self) -> int:
        return self._model.get_output_dim()
