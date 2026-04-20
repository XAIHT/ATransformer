"""
src/optimizer.py
================

Noam learning-rate schedule + Adam, from §5.3 of the paper — Equation (3).

Paper references
----------------
* §5.3 "Optimizer":

      > "We used the Adam optimizer [20] with β_1 = 0.9, β_2 = 0.98
      >  and ε = 10^{-9}.  We varied the learning rate over the course
      >  of training, according to the formula:

      >     lrate = d_model^{-0.5} · min(step_num^{-0.5},
      >                                  step_num · warmup_steps^{-1.5})  (3)

      >  This corresponds to increasing the learning rate linearly for
      >  the first warmup_steps training steps, and decreasing it
      >  thereafter proportionally to the inverse square root of the
      >  step number.  We used warmup_steps = 4000."

This module provides:

* :class:`NoamScheduler` — computes the scalar learning rate of Eq. (3).
* :func:`make_noam_optimizer` — builds the Adam optimizer with the
  paper-specified hyper-parameters and wraps it in a
  :class:`torch.optim.lr_scheduler.LambdaLR` driven by
  :class:`NoamScheduler`.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR


class NoamScheduler:
    """Callable implementing Eq. (3) of §5.3.

    The scheduler returns a *multiplier* (not an absolute learning
    rate), because :class:`torch.optim.lr_scheduler.LambdaLR` expects a
    function ``step → factor`` that is multiplied by the base LR of the
    optimizer.  To make the absolute LR coincide with Eq. (3) we set
    the optimizer's ``lr`` to 1.0 and fold the whole formula into the
    scheduler, exactly like the reference implementation.
    """

    def __init__(self, d_model: int, warmup_steps: int = 4000, factor: float = 1.0):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        # `factor` is a convenient way to globally re-scale the LR without
        # editing Eq. (3); the paper itself uses factor = 1.0.
        self.factor = factor

    def __call__(self, step: int) -> float:
        """Return Eq. (3) evaluated at ``step`` (1-indexed)."""
        # `step` must be ≥ 1 to avoid 0^-0.5 = ∞.  LambdaLR passes 0 on
        # the very first call, so we clamp.
        step = max(step, 1)
        # Eq. (3) of the paper, factor-scaled.
        return self.factor * (
            self.d_model ** -0.5
            * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        )


def make_noam_optimizer(model: torch.nn.Module,
                        d_model: int,
                        warmup_steps: int = 4000,
                        factor: float = 1.0):
    """Return ``(optimizer, lr_scheduler)`` wired exactly as §5.3 prescribes.

    * Adam β_1 = 0.9, β_2 = 0.98, ε = 1e-9         — §5.3.
    * Base LR = 1.0 (see :class:`NoamScheduler` docstring).
    * LR schedule: Eq. (3) of the paper, warmup_steps = 4000 by default.
    """
    optimizer = Adam(
        model.parameters(),
        lr=1.0,                          # the *real* LR comes from the scheduler
        betas=(0.9, 0.98),               # §5.3
        eps=1e-9,                        # §5.3
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=NoamScheduler(d_model=d_model,
                                warmup_steps=warmup_steps,
                                factor=factor),
    )
    return optimizer, scheduler
