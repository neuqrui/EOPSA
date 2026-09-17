"""No-op reward for pure OPSD distillation (no GRPO / no API judge)."""

REWARD_NAME = "opsd_noop"
REWARD_TYPE = "batch"


def compute_score(reward_inputs, **kwargs):
    """Return zero reward; distillation loss does not use this when outcome_ppo_coef=0."""
    return [{"overall": 0.0, "accuracy": 0.0, "format": 1.0} for _ in reward_inputs]
