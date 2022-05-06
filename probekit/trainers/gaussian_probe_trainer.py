from typing import List
import torch

from probekit.trainers.trainer import Trainer
from probekit.utils.types import PyTorchDevice, Specification
from probekit.utils.dataset import ClassificationDataset


# NOTE: Should be able to do this with Python 3.8. Consider upgrading
# GaussianProbeTrainerMode = Literal["mle", "map"]
GaussianProbeTrainerMode = str


class GaussianProbeTrainer(Trainer):
    def __init__(self, mode: GaussianProbeTrainerMode, dataset: ClassificationDataset,
                 device: PyTorchDevice):
        self._mode = mode

        # TODO: Allow custom priors
        self._map_data_driven_prior = True

        super().__init__(dataset=dataset, device=device)

    def _train(self):
        # Compute parameters
        means: List[torch.Tensor]
        covariance_matrices: List[torch.Tensor]
        class_probabilities: List[torch.Tensor]
        if self._mode == "mle":
            means, covariance_matrices, class_probabilities = self._train_mle()
        elif self._mode == "map":
            means, covariance_matrices, class_probabilities = self._train_map()
        else:
            raise Exception("Unknown Gaussian probe training mode.")

        self._means = means
        self._covariance_matrices = covariance_matrices
        self._class_probabilities = class_probabilities

    def _train_mle(self):
        means = []
        covariance_matrices = []
        class_counts = []
        for property_value, property_examples in self._dataset.items():
            # TODO: Optimize using new dataset commands
            embeddings = [ex.get_embedding() for ex in property_examples]
            embeddings_tensor = torch.stack(embeddings, dim=0).to(self.get_device())

            mean = embeddings_tensor.mean(dim=0)
            num_samples = embeddings_tensor.shape[0]
            covariance_matrix = (1 / num_samples) * (
                (embeddings_tensor - mean).T @ (embeddings_tensor - mean))

            means.append(mean)
            covariance_matrices.append(covariance_matrix)
            class_counts.append(len(property_examples))

        # Compute class probabilities
        total_examples = sum(class_counts)
        class_probabilities = torch.tensor([x / total_examples for x in class_counts])
        return means, covariance_matrices, class_probabilities

    def _train_map(self):
        means = []
        covariance_matrices = []
        class_counts = []
        for property_value, property_examples in self._dataset.items():
            embeddings = [ex.get_embedding() for ex in property_examples]
            embeddings_tensor = torch.stack(embeddings, dim=0).to(self.get_device())

            embeddings_mean = torch.mean(embeddings_tensor, dim=0)
            embeddings_scatter = (embeddings_tensor - embeddings_mean).T @ (embeddings_tensor - embeddings_mean)  # noqa

            d = embeddings_mean.shape[0]
            n = len(property_examples)

            if self._map_data_driven_prior:
                centered_embeddings = (embeddings_tensor - embeddings_mean)
                cov = centered_embeddings.t().matmul(centered_embeddings) / n

                self.Lambda = torch.diag(torch.diag(cov))
                self.nu = d + 2
                self.mu = embeddings_mean
                self.k = 0.01

            mu_update = (self.k * self.mu + n * embeddings_mean) / (self.k + n)

            k_update = self.k + n  # noqa
            nu_update = self.nu + n

            embeddings_scatter_prior = (embeddings_mean - self.mu) @ (embeddings_mean - self.mu).T
            Lambda_update = self.Lambda + embeddings_scatter
            Lambda_update += (self.k * n) / (self.k + n) * embeddings_scatter_prior

            # Get MAP estimates for mean and covariance
            mean = mu_update
            covariance_matrix = (nu_update + d + 2) ** -1 * Lambda_update

            means.append(mean)
            covariance_matrices.append(covariance_matrix)
            class_counts.append(len(property_examples))

        # Compute class probabilities
        total_examples = sum(class_counts)
        class_probabilities = torch.tensor([x / total_examples for x in class_counts])

        return means, covariance_matrices, class_probabilities

    def _get_specification(self, select_dimensions: List[int]) -> Specification:
        return {
            "class_probabilities": self._class_probabilities,
            "means": [x[select_dimensions] for x in self._means],
            "covariance_matrices": [x[select_dimensions].T[select_dimensions].T
                                    for x in self._covariance_matrices],
        }
