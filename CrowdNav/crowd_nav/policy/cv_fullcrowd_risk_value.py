"""Matched constant-velocity control for full-crowd risk correction."""

from crowd_nav.policy.bayesian_fullcrowd_risk_value import (
    BayesianFullCrowdRiskValuePolicy,
)
from crowd_nav.risk_models import ConstantVelocityRiskModel


class ConstantVelocityFullCrowdRiskValuePolicy(
    BayesianFullCrowdRiskValuePolicy
):
    """Use the proposed value correction with a deterministic CV predictor."""

    def __init__(self, config=None, device="cpu"):
        super().__init__(config=config, device=device)
        self.belief_filter = ConstantVelocityRiskModel(
            max_peds=self.belief_num_humans,
            modeled_peds=self.belief_num_humans,
            belief_dim=self.belief_k + 2,
        )


ConstantVelocityFullCrowdRiskValue = (
    ConstantVelocityFullCrowdRiskValuePolicy
)
