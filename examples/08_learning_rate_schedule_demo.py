"""
examples/08_learning_rate_schedule_demo.py
==========================================

Plot the Noam learning-rate schedule from §5.3 of the paper, Eq. (3):

    lrate = d_model^{-0.5} · min(step^{-0.5}, step · warmup_steps^{-1.5}).

Run
---
    python examples/08_learning_rate_schedule_demo.py

Outputs
-------
* Prints LR at a few key steps.
* Saves `noam_schedule.png` — the classic "linear warmup then inverse
  square-root decay" curve.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimizer import NoamScheduler


def main():
    # Reproduce Fig. 3-ish of the reference implementations of the paper,
    # for the "base" model (d_model = 512) and warmup_steps = 4000 (§5.3).
    sched = NoamScheduler(d_model=512, warmup_steps=4000)

    for step in [1, 100, 1000, 2000, 4000, 8000, 16000, 32000]:
        print(f"step {step:>6d}  ->  lr = {sched(step):.6e}")

    try:
        import matplotlib.pyplot as plt
        steps = list(range(1, 40_000))
        lrs   = [sched(s) for s in steps]

        plt.figure(figsize=(8, 4))
        plt.plot(steps, lrs)
        plt.xlabel("training step")
        plt.ylabel("learning rate")
        plt.title("Noam learning-rate schedule (§5.3, Eq. 3)  —  d_model=512, warmup=4000")
        plt.axvline(4000, color="red", linestyle="--", alpha=0.4, label="warmup_steps")
        plt.legend()
        out = os.path.join(os.path.dirname(__file__), "noam_schedule.png")
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        print(f"\nSaved LR curve to {out}")
    except Exception as exc:
        print(f"(matplotlib unavailable: {exc})")


if __name__ == "__main__":
    main()
