"""
Base Byzantine Attack Abstract Class.
Defines the interface for all adversarial parameter and data poisoning attacks
in smart grid federated learning.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. Attacks must remain strictly isolated from normal client training routines.
2. An unusual update is NOT proof of malicious intent (non-IID industrial or rural
   substations naturally exhibit high variance, peak loads, and unique gradient trajectories).
3. Ground-truth attacker identities are recorded exclusively for post-hoc research evaluation;
   server-side aggregation strategies and trust detectors MUST NEVER access ground-truth labels.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class BaseByzantineAttack(ABC):
    """
    Abstract base class for all Byzantine attack vectors.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, seed: int = 42):
        self.name = name
        self.config = config or {}
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def perturb_parameters(
        self,
        global_parameters: List[np.ndarray],
        local_parameters: List[np.ndarray],
        client_id: str,
        server_round: int,
    ) -> List[np.ndarray]:
        """
        Apply Byzantine parameter perturbation.

        Parameters
        ----------
        global_parameters : List[np.ndarray]
            Global model parameter arrays broadcast by the server at start of round.
        local_parameters : List[np.ndarray]
            Honest model parameter arrays computed by the local client gradient descent.
        client_id : str
            Identifier of the client executing the attack.
        server_round : int
            Current federated learning round.

        Returns
        -------
        List[np.ndarray]
            Poisoned/perturbed parameter arrays to be transmitted to the server.
        """
        pass
