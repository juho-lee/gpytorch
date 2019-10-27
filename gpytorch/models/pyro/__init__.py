#!/usr/bin/env python3

try:
    from .pyro_gp import PyroGP
    from .trace_predictive_log_likelihood import Trace_PredictiveLogLikelihood
except ImportError:
    class PyroGP(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Cannot use a PyroGP because you dont have Pyro installed.")

    class Trace_PredictiveLogLikelihood(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Cannot use a Trace_PredictiveLogLikelihood because you dont have Pyro installed.")


__all__ = [
    "PyroGP",
    "Trace_PredictiveLogLikelihood",
]
