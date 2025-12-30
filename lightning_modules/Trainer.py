from lightning.pytorch import Trainer
from typing import Any, Dict, Generator, Iterable, List, Optional, Union
import logging
from .EvaluationLoop import customEvaluationLoop
from .TrainingEpochLoop import customTrainingEpochLoop
from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from lightning.pytorch.utilities.argparse import _defaults_from_env_vars
log = logging.getLogger(__name__)

class customTrainer(Trainer):
    @_defaults_from_env_vars
    def __init__(
        self,
        *,
        max_steps: int = -1,
        min_steps: Optional[int] = None,
        inference_mode: bool = True,
        **kwargs
    ) -> None:
        r"""Customize every aspect of training via flags.

        Args:
            inference_mode: Whether to use :func:`torch.inference_mode` or :func:`torch.no_grad` during
                evaluation (``validate``/``test``/``predict``).

        Raises:
            TypeError:
                If ``gradient_clip_val`` is not an int or float.

            MisconfigurationException:
                If ``gradient_clip_algorithm`` is invalid.

        """
        super().__init__(inference_mode=inference_mode, **kwargs)
        print('customTrainer')
        self.fit_loop.epoch_loop = customTrainingEpochLoop(self, min_steps=min_steps, max_steps=max_steps)
        self.validate_loop = customEvaluationLoop(
            self, TrainerFn.VALIDATING, RunningStage.VALIDATING, inference_mode=inference_mode
        )
        self.test_loop = customEvaluationLoop(self, TrainerFn.TESTING, RunningStage.TESTING, inference_mode=inference_mode)
