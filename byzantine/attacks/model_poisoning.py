"""
Targeted Model Poisoning Byzantine Attack.
Injects structured adversarial Gaussian perturbations into model weights
to corrupt theft detection decision boundaries without causing trivial zero gradients.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from byzantine.attacks.base_attack import BaseByzantineAttack


class ModelPoisoningAttack(BaseByzantineAttack):
    """
    Model Poisoning Attack.
    Adds targeted Gaussian noise scaled by adversarial boost factor:
    w_poisoned = w_local + adversarial_boost * Normal(noise_mean, noise_std * layer_std)
    """

    def __init__(
        self,
        noise_mean: float = 0.0,
        noise_std: float = 1.5,
        adversarial_boost: float = 2.0,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        super().__init__(name="model_poisoning", config=config, seed=seed)
        self.noise_mean = float(self.config.get("noise_mean", noise_mean))
        self.noise_std = float(self.config.get("noise_std", noise_std))
        self.adversarial_boost = float(self.config.get("adversarial_boost", adversarial_boost))

    def perturb_parameters(
        self,
        global_parameters: List[np.ndarray],
        local_parameters: List[np.ndarray],
        client_id: str,
        server_round: int,
    ) -> List[np.ndarray]:
        """
        Inject targeted adversarial noise into local weights.
        """
        poisoned_params = []
        for w_loc in local_parameters:
            std_scale = float(np.std(w_loc)) if np.std(w_loc) > 1e-6 else 1.0
            noise = self.rng.normal(
                loc=self.noise_mean,
                scale=self.noise_std * std_scale,
                size=w_loc.shape,
            )
            perturbed = w_loc + (self.adversarial_boost * noise)
            poisoned_params.append(perturbed.astype(w_loc.dtype))
        return poisoned_params
