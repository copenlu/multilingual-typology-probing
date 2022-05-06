from typing import List, Optional
import torch
import torch.optim as optim
from tqdm import trange

from probekit.trainers.trainer import Trainer
from probekit.utils.types import PyTorchDevice, Specification
from probekit.utils.dataset import FastTensorDataLoader, ClassificationDataset
from probekit.models.discriminative.neural_probe import NeuralProbeModel, MaskAndMapWrapper


class NeuralProbeTrainer(Trainer):
    def __init__(
            self, model: NeuralProbeModel, dataset: ClassificationDataset, device: PyTorchDevice,
            decomposable: bool = False, lr: float = 1e-2, num_epochs: int = 2000, epsilon: float = 1e-5,
            batch_size: Optional[int] = None, report_progress: bool = True):
        self._model = model
        self._decomposable = decomposable
        self._lr = lr
        self._num_epochs = num_epochs
        self._epsilon = epsilon
        self._batch_size = batch_size or 256
        self._report_progress = report_progress

        super().__init__(dataset=dataset, device=device)

    def _train(self):
        # We only pretrain if we have forced the neural probe to be decomposable
        if self._decomposable:
            self._train_for_dimensions()

    def _train_for_dimensions(self, select_dimensions: Optional[List[int]] = None):
        inputs_tensor, values_tensor = self._dataset.get_inputs_values_tensor()
        num_samples, dim = inputs_tensor.shape
        num_values = len(self.get_property_values())

        # If select_dimensions is set, we want to train the probe on a specific set of dimensions
        if select_dimensions:
            mask = torch.zeros(dim).scatter(0, torch.tensor(select_dimensions), 1.0).to(self.get_device())
        else:
            mask = torch.ones(dim).to(self.get_device())

        model = self._model
        model.reset_parameters()

        # Train model
        optimizer = optim.Adam(model.parameters(), lr=self._lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        data_loader = FastTensorDataLoader(
            inputs_tensor, values_tensor, batch_size=self._batch_size, shuffle=True)

        t = trange(self._num_epochs, desc="Training neural probe", disable=not self._report_progress,
                   leave=False)
        loss_prev = float('inf')
        for epoch in t:
            for minibatch, minibatch_assignment in data_loader:
                minibatch_size = minibatch.shape[0]

                class_scores = model(minibatch * mask)
                assert class_scores.shape == (minibatch_size, num_values)

                loss = loss_fn(class_scores, minibatch_assignment)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if abs(loss_prev - loss.cpu().item()) <= self._epsilon:
                # We stop when loss doesn't decrease anymore
                break

            loss_prev = loss.cpu().item()
            t.set_postfix(loss=loss_prev)

    def _get_specification(self, select_dimensions: List[int]) -> Specification:
        # If we have not forced the neural probe to be decomposable, we must train it from scratch
        # for every new selection of dimensions
        if not self._decomposable:
            self._train_for_dimensions(select_dimensions)

        # Apply a masking and mapping layer to the model's inputs.
        masked_model = MaskAndMapWrapper(self._model, select_dimensions)

        return {
            "model": masked_model
        }
