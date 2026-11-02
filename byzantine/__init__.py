"""
byzantine package.
Blockchain-Based Byzantine-Resilient Federated Learning for Smart Grid Energy Theft Detection.

Modules:
- attack_manager: ByzantineAttackManager coordinating malicious nodes, scheduling, and attack dispatching.
- attacks: Implementations of sign-flipping, model poisoning, random updates, and label flipping.
"""

from byzantine.attack_manager import (
    ANOMALY_EPISTEMIC_DISCLAIMER,
    ByzantineAttackManager,
)
from byzantine.attacks import (
    BaseByzantineAttack,
    LabelFlippingAttack,
    ModelPoisoningAttack,
    RandomUpdateAttack,
    SignFlippingAttack,
)

__all__ = [
    "ByzantineAttackManager",
    "BaseByzantineAttack",
    "SignFlippingAttack",
    "ModelPoisoningAttack",
    "RandomUpdateAttack",
    "LabelFlippingAttack",
    "ANOMALY_EPISTEMIC_DISCLAIMER",
]
