"""
Sign-Flipping Byzantine Attack.
Inverts the gradient/parameter update direction to drive the global federated model
away from the optimal theft detection decision boundary.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from byzantine.attacks.base_attack import BaseByzantineAttack


class SignFlippingAttack(BaseByzantineAttack):
    """
    Sign-Flipping Attack.
    Computes delta Delta_w = w_local - w_global and returns:
    w_poisoned = w_global + (multiplier * scale_factor) * Delta_w
    Default: multiplier = -1.0, scale_factor = 1.5.
    """

    def __init__(
        self,
        multiplier: float = -1.0,
        scale_factor: float = 1.5,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        super().__init__(name="sign_flipping", config=config, seed=seed)
        self.multiplier = float(self.config.get("multiplier", multiplier))
        self.scale_factor = float(self.config.get("scale_factor", scale_factor))
        self.effective_scalar = self.multiplier * self.scale_factor

    def perturb_parameters(
        self,
        global_parameters: List[np.ndarray],
        local_parameters: List[np.ndarray],
        client_id: str,
        server_round: int,
    ) -> List[np.ndarray]:
        """
        Invert and scale the local update direction.
        """
        poisoned_params = []
        for w_glob, w_loc in zip(global_parameters, local_parameters):
            delta = w_loc - w_glob
            poisoned = w_glob + (self.effective_scalar * delta)
            poisoned_params.append(poisoned.astype(w_loc.dtype))
        return poisoned_params
