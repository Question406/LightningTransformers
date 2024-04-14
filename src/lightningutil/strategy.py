from logging import WARN
from typing import Any, Dict, List
from pytorch_lightning.plugins.precision import Precision
from pytorch_lightning.strategies import DeepSpeedStrategy
from typing import Optional
from lightning.fabric.utilities.types import _PATH, LRScheduler, ReduceLROnPlateau
from typing_extensions import override

class NoIntermediateStatesDeepSpeedStrategy(DeepSpeedStrategy):
    @override
    def save_checkpoint(self, checkpoint: Dict, filepath: _PATH, storage_options: Optional[Any] = None) -> None:
        """ Override default save_checkpoint method to avoid deep speed intermediate states
        """
        if self.is_global_zero:
            self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)

