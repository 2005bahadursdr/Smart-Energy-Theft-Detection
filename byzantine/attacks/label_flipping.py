"""
Label-Flipping Byzantine Attack.
Inverts training labels on compromised client nodes (Theft 1 -> 0, Normal 0 -> 1)
to subtly compromise the theft classification boundary from the data layer.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from byzantine.attacks.base_attack import BaseByzantineAttack
from fl_system.models.dataset import MeterSequenceDataset


class LabelFlippingAttack(BaseByzantineAttack):
    """
    Label-Flipping Attack.
    Inverts local ground-truth labels (1 -> 0, 0 -> 1) during local model training.
    """

    def __init__(
        self,
        flip_ratio: float = 1.0,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        super().__init__(name="label_flipping", config=config, seed=seed)
        self.flip_ratio = float(self.config.get("flip_ratio", flip_ratio))

    def flip_dataset(self, dataset: MeterSequenceDataset) -> MeterSequenceDataset:
        """
        Flip binary labels of a local MeterSequenceDataset.
        Strictly isolated to compromised substation training partitions.
        """
        if isinstance(dataset.y, torch.Tensor):
            y_orig = dataset.y.cpu().numpy().copy()
        else:
            y_orig = np.asarray(dataset.y).copy()

        n_samples = len(y_orig)
        if n_samples == 0:
            return dataset

        n_flip = int(n_samples * self.flip_ratio)
        flip_indices = self.rng.choice(n_samples, size=n_flip, replace=False)

        y_flipped = y_orig.copy()
        # Binary flip: 0 -> 1, 1 -> 0
        y_flipped[flip_indices] = 1.0 - y_flipped[flip_indices]

        X_tensor = dataset.X if isinstance(dataset.X, torch.Tensor) else torch.tensor(dataset.X)
        y_tensor = torch.tensor(y_flipped, dtype=torch.float32)

        return MeterSequenceDataset(
            X=X_tensor,
            y=y_tensor,
            meter_ids=list(dataset.meter_ids),
            timestamps=list(dataset.timestamps),
            is_synthetic=dataset.is_synthetic,
            feature_names=getattr(dataset, "feature_names", None),
        )

    def perturb_parameters(
        self,
        global_parameters: List[np.ndarray],
        local_parameters: List[np.ndarray],
        client_id: str,
        server_round: int,
    ) -> List[np.ndarray]:
        """
        Parameter-level fallback for label flipping: inverts the output classification weights.
        """
        poisoned_params = []
        for i, (w_loc, w_glob) in enumerate(zip(local_parameters, global_parameters)):
            # Invert final classifier layer weights and biases (last 2 parameter tensors)
            if i >= len(local_parameters) - 2:
                delta = w_loc - w_glob
                poisoned = w_glob - delta
                poisoned_params.append(poisoned.astype(w_loc.dtype))
            else:
                poisoned_params.append(w_loc.copy())
        return poisoned_params
