from typing import Optional, List, Dict, Any, Union
import torch
import torch.nn as nn
import torch_struct

from .flexible_neural_probe_trainer import FlexibleNeuralProbeTrainer


class ConditionalPoissonVariationalFamily(nn.Module):
    def __init__(self, D: int, K: int):
        super().__init__()

        self.D = D
        self.K = K
        self.weights = nn.Parameter(torch.rand(self.D))

    def get_device(self):
        return self.weights.device

    def run_on_semiring(self, sr, cache):
        weights_sr = sr.convert(self.weights)
        return sr.unconvert(self.S_N_n(sr, weights_sr, self.D, self.K, cache))

    def entropy(self, cache=None):
        return self.run_on_semiring(torch_struct.EntropySemiring, cache if cache is not None else [])

    def logZ(self, cache=None):
        return self.run_on_semiring(torch_struct.LogSemiring, cache if cache is not None else [])

    def logprob(self, mask, logZ=None):
        if logZ is None:
            logZ = self.logZ()

        return mask.float() @ self.weights - logZ

    def sample(self, num_samples, log_cache):
        num_to_sample = torch.ones(num_samples, device=self.get_device(), dtype=torch.long)
        return self._sample(num_samples, num_to_sample, log_cache)

    def get_from_cache(self, N, n, cache):
        return cache[N - 1][..., n]

    def _sample(self, num_samples, num_to_sample, log_cache):
        """
        Conditional Poisson sampling of num_sample masks with exactly num_to_sample dimensions active on
        each sampled mask.
        """
        samples = torch.zeros(num_samples, self.D, device=self.weights.device, dtype=torch.long)
        for d in reversed(range(1, self.D + 1)):
            num_sampled_dimensions = samples.sum(dim=1)
            num_dimensions_to_sample = num_to_sample - num_sampled_dimensions
            assert num_sampled_dimensions.shape == (num_samples,)

            # Compute coin flip probabilities as if we choose to sample the dimension in all cases
            sample_dim_prob = (self.weights[d - 1]
                               + self.get_from_cache(d - 1, num_dimensions_to_sample - 1, log_cache)
                               - self.get_from_cache(d, num_dimensions_to_sample, log_cache)).exp()

            # In these cases, we are forced to sample everything left. Hence, force sampling this dim.
            sample_dim_prob.masked_fill_(num_dimensions_to_sample >= d, 1.)

            # In these case, we sampled all we needed to sample already. Hence, force NOT sampling of dim.
            sample_dim_prob.masked_fill_(num_sampled_dimensions >= num_to_sample, 0.)

            # Compute samples
            samples[:, d - 1] = torch.bernoulli(sample_dim_prob)

        assert (samples.sum(dim=1) == num_to_sample).all()
        return samples

    def S_N_n(self, sr, weights, N, n, cache):
        assert len(cache) == 0, "Cache is recreated when this is called"

        one = sr.convert(torch.tensor(0., device=self.get_device()))
        sr.one_(one)
        zero = sr.convert(torch.tensor(0., device=self.get_device()))
        sr.zero_(zero)
        zero_t = sr.convert(torch.zeros(1, device=self.get_device()))
        sr.zero_(zero_t)

        # Need to add initial items
        start = sr.convert(torch.zeros(N + 1, device=self.get_device()))
        sr.zero_(start)
        start[..., 1 - 1] = one
        start[..., 2 - 1] = weights[..., 0]
        cache += [start]

        for d in range(1, N):
            S_less_curr = cache[d - 1]
            S_less_less = torch.cat([zero_t, S_less_curr[..., :-1]], dim=-1)

            assert S_less_less.shape == S_less_curr.shape

            S_curr = sr.plus(sr.mul(weights[..., d], S_less_less), S_less_curr)
            cache += [S_curr]

        assert len(cache) == N

        return cache[N - 1][..., n]


class UniformPoissonVariationalFamily(ConditionalPoissonVariationalFamily):
    """
    First sample a number of dimensions to sample uniformly, and then sample a mask containing that number
    of dimensions.
    """
    def __init__(self, D):
        super().__init__(D, D)

    def entropy(self, cache=None):
        # Populate cache
        cache = cache if cache is not None else []
        sr = torch_struct.EntropySemiring
        self.run_on_semiring(sr, cache)

        # ::  Entropy ::
        # Need to compute entropy of p(C, k) = p(k) p(C | k)
        # H(C, K) = H(K) + H(C | K)
        # where H(C | K) = \sum_k p(K = k) H(C | K = k)

        # H(K)
        ent_uniform = torch.tensor(self.D, device=self.get_device()).log()

        # H(C | K)
        ent_cps = sr.unconvert(cache[self.D - 1])
        assert ent_cps.shape == (self.D + 1,)
        ent_cp = ent_cps[1:].mean()

        return ent_uniform + ent_cp

    def logZ(self, cache=None):
        # Populate cache
        cache = cache if cache is not None else []
        sr = torch_struct.LogSemiring
        self.run_on_semiring(sr, cache)

        # :: Normalizer ::
        normalizers_cps = sr.unconvert(cache[self.D - 1])
        assert normalizers_cps.shape == (self.D + 1,)

        normalizer_uniform = torch.tensor(self.D, device=self.get_device()).log()
        return normalizers_cps[1:].logsumexp(dim=0) - normalizer_uniform

    def sample(self, num_samples, log_cache):
        num_to_sample = torch.randint(1, self.D + 1, (num_samples,), device=self.get_device())
        return self._sample(num_samples, num_to_sample, log_cache)


class ConditionalPoissonTrainer(FlexibleNeuralProbeTrainer):
    def __init__(self, mc_samples: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mc_samples = mc_samples
        self.active_neurons = 300
        self.variational_family = UniformPoissonVariationalFamily(
            self._dataset.get_dimensionality()).to(self.get_device())

        self.entropy_scale = torch.tensor([0.001], device=self.get_device())

    def setup_parameter_list(self):
        params_to_optimize = super().setup_parameter_list()
        params_to_optimize.extend(self.variational_family.parameters())

        return params_to_optimize

    def train_loss(self, minibatch: torch.Tensor, minibatch_assignment: torch.Tensor,
                   select_dimensions: Optional[List[int]], others: Dict[str, Any]):
        minibatch_size, dim = minibatch.shape
        num_values = len(self.get_property_values())
        model = self._model
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        log_cache: List[torch.Tensor] = []
        logZ = self.variational_family.logZ(cache=log_cache)

        # If select_dimensions is set, we want to train the probe statically on a specific set of dimensions
        # (as it isn't decomposable).
        if select_dimensions:
            # NOTE: This case shouldn't arise unless we try to train with manual mode.
            raise NotImplementedError("Need to think about how to handle this case, if it ever arises.")

        # COMPUTE FIRST TERM OF LOSS, AKA., THE MC ESTIMATE OF THE LIKELIHOOD UNDER Q_\VPHI
        # Note that in practice we compute the NEGATIVE likelihood as that is more convenient
        loss_mc = torch.zeros(minibatch_size, device=self.get_device())
        for _ in range(self.mc_samples):
            # Sample a mask from the variational distribution
            mask = self.variational_family.sample(minibatch_size, log_cache=log_cache).detach()

            # Compute model's predicted probabilities
            class_scores = model(minibatch, mask)
            assert class_scores.shape == (minibatch_size, num_values)

            # Compute negative log-likelihood
            loss_mc += loss_fn(class_scores, minibatch_assignment)

        # Average each loss across samples
        loss_mc /= self.mc_samples

        # Obtain expected loss per datapoint
        loss_mc = loss_mc.sum() / minibatch_size

        # :::::::: REINFORCE ::::::::
        loss_rf = torch.zeros(minibatch_size, device=self.get_device())
        loss_rf_real = torch.zeros(minibatch_size, device=self.get_device())
        for _ in range(self.mc_samples):
            # Sample a mask from the variational distribution
            mask = self.variational_family.sample(minibatch_size, log_cache=log_cache).detach()

            # Compute model's predicted probabilities
            class_scores = model(minibatch, mask)
            assert class_scores.shape == (minibatch_size, num_values)

            # Compute negative log-likelihood
            loss_rf += loss_fn(
                class_scores, minibatch_assignment).detach() * self.variational_family.logprob(
                    mask, logZ=logZ)
            loss_rf_real += loss_fn(class_scores, minibatch_assignment).detach()

        # Average each loss across samples
        loss_rf /= self.mc_samples
        loss_rf_real /= self.mc_samples

        # Obtain expected loss per datapoint
        loss_rf = loss_rf.sum() / minibatch_size
        loss_rf_real = loss_rf_real.mean()

        # COMPUTE SECOND TERM OF LOSS, AKA., ENTROPY
        # We make it negative since we are minimizing, not maximizing, the objective
        loss_entropy = -self.variational_family.entropy()

        # COMPUTE REGULARIZATION
        l1_weight, l2_weight = self._l1_weight, self._l2_weight

        weights_regularization: Union[torch.Tensor, float] = 0.0
        for p in model.parameters():
            weights_regularization += l1_weight * p.abs().sum() + l2_weight * (p ** 2).sum()

        loss = loss_mc + loss_entropy + loss_rf + weights_regularization

        # The actual likelihood
        loss_real = loss_mc + loss_entropy + weights_regularization

        # The first loss is used for backpropagation.
        # The second loss is the "correct" one, used for early stopping.

        return loss, loss_real
