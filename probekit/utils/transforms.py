from overrides import overrides
import torch

from probekit.utils.dataset import DatasetTransform, ClassificationDataset
from probekit.utils.types import PyTorchDevice


class PCATransform(DatasetTransform):
    def __init__(self, mean: torch.Tensor, projection: torch.Tensor):
        super().__init__()

        self._mean = mean.detach()
        self._projection = projection.detach()

    def to(self, device: PyTorchDevice):
        self._mean = self._mean.to(device)
        self._projection = self._projection.to(device)
        return self

    @classmethod
    def from_dataset(cls, dataset: ClassificationDataset, num_components: int):
        embeddings_tensor_concat = dataset.get_inputs_values_tensor()[0]

        dims = embeddings_tensor_concat.shape[1]
        assert num_components <= dims, "Can't select more principal components than there are dimensions."
        assert num_components > 0, "At least 1 dimension needs to be selected."

        # Center
        N, d = embeddings_tensor_concat.shape
        mean = embeddings_tensor_concat.mean(dim=0)
        assert mean.shape == (d,)

        centered = embeddings_tensor_concat - mean
        unscaled_cov = centered.T @ centered
        eigenvalues, eigenvectors = torch.symeig(unscaled_cov, eigenvectors=True)
        assert eigenvectors.shape == (d, d)

        projection = eigenvectors[:, -num_components:]
        return cls(mean, projection)

    @overrides
    def transform(self, input: torch.Tensor) -> torch.Tensor:
        return (input - self._mean) @ self._projection
