"""
byzantine.attacks package.
Adversarial update generators for Byzantine-resilient federated learning research.
"""

from byzantine.attacks.base_attack import BaseByzantineAttack
from byzantine.attacks.label_flipping import LabelFlippingAttack
from byzantine.attacks.model_poisoning import ModelPoisoningAttack
from byzantine.attacks.random_update import RandomUpdateAttack
from byzantine.attacks.sign_flipping import SignFlippingAttack

__all__ = [
    "BaseByzantineAttack",
    "SignFlippingAttack",
    "ModelPoisoningAttack",
    "RandomUpdateAttack",
    "LabelFlippingAttack",
]
