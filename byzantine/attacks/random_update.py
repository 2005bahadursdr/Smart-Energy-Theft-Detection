"""
Random Update (Noise Injection) Byzantine Attack.
Completely replaces local model updates with uninformative, high-variance Gaussian noise.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from byzantine.attacks.base_attack import BaseByzantineAttack


class RandomUpdateAttack(BaseByzantineAttack):
    """
    Random Update Attack.
    Substitutes legitimate local parameters with random Gaussian vectors:
    w_poisoned ~ Normal(noise_mean, noise_std)
    """

    def __init__(
        self,
        noise_mean: float = 0.0,
        noise_std: float = 3.0,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        super().__init__(name="random_update", config=config, seed=seed)
        self.noise_mean = float(self.config.get("noise_mean", noise_mean))
        self.noise_std = float(self.config.get("noise_std", noise_std))

    def perturb_parameters(
        self,
        global_parameters: List[np.ndarray],
        local_parameters: List[np.ndarray],
        client_id: str,
        server_round: int,
    ) -> List[np.ndarray]:
        """
        Replace legitimate local weights with purely random Gaussian noise.
        """
        poisoned_params = []
        for w_loc in local_parameters:
            noise = self.rng.normal(
                loc=self.noise_mean,
                scale=self.noise_std,
                size=w_loc.shape,
            )
            poisoned_params.append(noise.astype(w_loc.dtype))
        return poisoned_params
