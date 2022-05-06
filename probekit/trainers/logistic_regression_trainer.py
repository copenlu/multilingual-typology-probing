from typing import List, Optional
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from probekit.trainers.trainer import Trainer
from probekit.utils.types import PyTorchDevice, Specification
from probekit.utils.dataset import ClassificationDataset


class LogisticRegressionTrainer(Trainer):
    def __init__(
            self, dataset: ClassificationDataset, device: PyTorchDevice,
            lr: float = 1e-2, num_epochs: int = 2000, batch_size: Optional[int] = None):
        self._lr = lr
        self._num_epochs = num_epochs
        self._epsilon = 1e-7
        self._batch_size = batch_size or 1000000

        super().__init__(dataset=dataset, device=device)

    def _train(self):
        # We cannot pretrain LogisticRegression
        pass

    def _train_for_dimensions(self, select_dimensions: List[int]):
        inputs_tensor, values_tensor = self._dataset.get_inputs_values_tensor(select_dimensions)
        num_samples, dim = inputs_tensor.shape
        num_values = len(self.get_property_values())

        weights = torch.randn((dim, num_values), requires_grad=True, device=self.get_device())
        biases = torch.randn((num_values,), requires_grad=True, device=self.get_device())

        optimizer = optim.Adam([weights, biases], lr=self._lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        dataset = TensorDataset(inputs_tensor, values_tensor)
        data_loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        t = trange(self._num_epochs, desc="Training logistic regression", disable=False)
        loss_prev = float('inf')
        for epoch in t:
            for minibatch, minibatch_assignment in data_loader:
                minibatch_size = minibatch.shape[0]

                class_scores = minibatch @ weights + biases
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

        return weights.detach(), biases.detach()

    def _get_specification(self, select_dimensions: List[int]) -> Specification:
        weights, biases = self._train_for_dimensions(select_dimensions)

        return {
            "weights": weights,
            "biases": biases,
        }
