from overrides import overrides
import torch

from probekit.models.probe import Probe


class GenerativeProbe(Probe):
    r"""
    Defines a general generative probing model, which defines a joint distribution by means of factoring
    it as :math:`p(h, \pi) = p(h \mid \pi) p(\pi)`.
    Doing so allows the evaluation of :math:`p(\pi \mid h)`, which is required for any probe.
    """
    def _log_prob_input_given_class(self, inputs: torch.Tensor):
        r"""Returns a tensor of log probabilities for :math:`\log p(h \mid \pi)` associated with a set of
        inputs for every possible class.

        Returns a tensor of log probabilities for :math:`\log p(h \mid \pi)` associated with a set of inputs
        for every possible class.
        Note that this is different from the method exposed by :class:`Probe`, which defines the
        discriminative probabilities directly, :math:`\log p(\pi \mid h)`.

        Args:
            inputs (torch.Tensor): An :math:`N \times d` tensor, where :math:`N` is the number of datapoints
                and :math:`d` is the dimensionality of the datapoints.

        Returns:
            torch.Tensor: An :math:`N \times K` tensor with the log probability of sampling that each
                datapoint from a class.
        """
        raise NotImplementedError

    def _log_prob_class(self):
        r"""Returns a tensor with the log probabilities of sampling each class, for :math:`\log p(\pi)`.

        Returns a tensor of log probabilities of sampling a given class, for :math:`\log p(\pi)`.
        If `classes` is provided, this returns a vector of log probabilities only for the classes provided
        for each input.

        Args:
            classes (torch.Tensor, optional): If provided, it should be an :math:`N`-dimensional vector with
                the classes for each datapoint. Defaults to None.

        Returns:
            torch.Tensor: An :math:`K`-dimensional of log probabilities associated with sampling a class.
        """
        raise NotImplementedError

    @overrides
    def _log_prob_class_given_input(self, inputs: torch.Tensor):
        r"""Implements of :math:`\log p(\pi \mid h)` based on the factored joint distribution.

        Args:
            inputs (torch.Tensor): An :math:`N \times d` tensor, where :math:`N` is the number of datapoints
                and :math:`d` is the dimensionality of the datapoints.

        Returns:
            torch.Tensor: An :math:`N \times K` tensor of the log probabilities for each possible
                classification.
        """
        log_probs_joint = self._log_prob_class().unsqueeze(0) + self._log_prob_input_given_class(inputs)
        log_probs_normalizer = log_probs_joint.logsumexp(dim=1).reshape(-1, 1)
        return log_probs_joint - log_probs_normalizer
