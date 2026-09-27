"""Fiber, amplifier, WDM, and optical-network channel models."""

from physics_nlo_eval.channel.assessment import (
    FiberSpan,
    NLIChannel,
    NonlinearityResult,
    assess_fiber_nonlinearity,
    fiber_nonlinearity_assessment,
)
from physics_nlo_eval.channel.fiber import EDFA, FiberChannel, IdealAmplifier
from physics_nlo_eval.channel.network import ROADM, PumpSpec, RamanPump
from physics_nlo_eval.channel.wdm import ChannelExtractor, ChannelSpec, WDMDemux, WDMMux, WDMPlan

__all__ = [
    "ChannelExtractor",
    "ChannelSpec",
    "EDFA",
    "FiberChannel",
    "FiberSpan",
    "IdealAmplifier",
    "NLIChannel",
    "NonlinearityResult",
    "PumpSpec",
    "ROADM",
    "RamanPump",
    "WDMDemux",
    "WDMMux",
    "WDMPlan",
    "assess_fiber_nonlinearity",
    "fiber_nonlinearity_assessment",
]
