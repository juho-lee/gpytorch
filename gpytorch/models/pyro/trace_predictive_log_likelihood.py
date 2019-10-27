#!/usr/bin/env python3

import math
from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace
from pyro.infer.util import MultiFrameTensor, get_plate_stacks, is_validation_enabled, torch_item
from pyro.util import check_if_enumerated, warn_if_nan


def _compute_log_r(model_trace, guide_trace):
    log_r = MultiFrameTensor()
    stacks = get_plate_stacks(model_trace)
    for name, model_site in model_trace.nodes.items():
        if model_site["type"] == "sample":
            log_r_term = model_site["log_prob"]
            if not model_site["is_observed"]:
                log_r_term = log_r_term - guide_trace.nodes[name]["log_prob"]
            log_r.add((stacks[name], log_r_term.detach()))
    return log_r


class Trace_PredictiveLogLikelihood(ELBO):
    """
    A trace implementation of predictive log likelihood-based inference.
    This is similar to Pyro's Trace_ELBO, but where the ELBO is replaced by
    the predictive log likelihood proposed by `Jankowiak et al., 2019`_.

    .. note::

        This loss function requires particles are vectorized.

    .. _Jankowiak et al., 2019:
        http://bit.ly/predictive_gp
    """
    def __init__(self, num_particles=1, vectorize_particles=True, max_plate_nesting=float('inf'), retain_graph=None):
        if not vectorize_particles:
            raise ValueError("Trace_PredictiveLogLikelihood requires that vectorize_particles=True")
        super().__init__(
            num_particles=num_particles,
            vectorize_particles=True,
            max_plate_nesting=max_plate_nesting,
            retain_graph=retain_graph
        )

    def _get_trace(self, model, guide, *args, **kwargs):
        """
        Returns a single trace from the guide, and the model that is run against it.
        """
        model_trace, guide_trace = get_importance_trace("flat", self.max_plate_nesting, model, guide, *args, **kwargs)
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the predictive log likelihood
        :rtype: float

        Evaluates the predictive log likelihood with an estimator that uses num_particles many samples/particles.
        """
        loss = self.differentiable_loss(model, guide, *args, **kwargs)
        return torch_item(loss)

    def _differentiable_loss_particle(self, model_trace, guide_trace):
        elbo_particle = 0
        surrogate_elbo_particle = 0
        log_r = None

        # compute elbo and surrogate elbo
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                # This here is where the code differs from Trace_ELBO
                # Rather than computing log_prob_sum for the observed variables
                # we instead compute a logsumexp over the sampel dimension
                if site["is_observed"]:
                    num_samples = site["log_prob"].size(0)
                    log_prob_sum = site["log_prob"].add(-math.log(num_samples)).logsumexp(dim=0)
                    log_prob_sum = log_prob_sum.mul(num_samples).sum()
                    # print("diffs", log_prob_sum, site["log_prob"].sum(), site["log_prob_sum"])
                else:
                    log_prob_sum = site["log_prob_sum"]
                elbo_particle = elbo_particle + torch_item(log_prob_sum)
                surrogate_elbo_particle = surrogate_elbo_particle + log_prob_sum

        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample":
                log_prob, score_function_term, entropy_term = site["score_parts"]

                elbo_particle = elbo_particle - torch_item(site["log_prob_sum"])

                if not is_identically_zero(entropy_term):
                    surrogate_elbo_particle = surrogate_elbo_particle - entropy_term.sum()

                if not is_identically_zero(score_function_term):
                    if log_r is None:
                        log_r = _compute_log_r(model_trace, guide_trace)
                    site = log_r.sum_to(site["cond_indep_stack"])
                    surrogate_elbo_particle = surrogate_elbo_particle + (site * score_function_term).sum()

        return -elbo_particle, -surrogate_elbo_particle

    def differentiable_loss(self, model, guide, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the model and guide parameters
        """
        loss = 0.
        surrogate_loss = 0.
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            loss_particle, surrogate_loss_particle = self._differentiable_loss_particle(model_trace, guide_trace)
            surrogate_loss += surrogate_loss_particle / self.num_particles
            loss += loss_particle / self.num_particles
        warn_if_nan(surrogate_loss, "loss")
        return loss + (surrogate_loss - torch_item(surrogate_loss))

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the predictive log likelihood
        :rtype: float

        Computes the predictive log likelihood as well as the surrogate PLL
        that is used to form the gradient estimator.  Performs backward on the
        latter. Num_particle many samples are used to form the estimators.
        """
        loss = 0.0
        # grab a trace from the generator
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            loss_particle, surrogate_loss_particle = self._differentiable_loss_particle(model_trace, guide_trace)
            loss += loss_particle / self.num_particles

            # collect parameters to train from model and guide
            trainable_params = any(
                site["type"] == "param" for trace in (model_trace, guide_trace)
                for site in trace.nodes.values()
            )

            if trainable_params and getattr(surrogate_loss_particle, 'requires_grad', False):
                surrogate_loss_particle = surrogate_loss_particle / self.num_particles
                surrogate_loss_particle.backward(retain_graph=self.retain_graph)
        warn_if_nan(loss, "loss")
        return loss
