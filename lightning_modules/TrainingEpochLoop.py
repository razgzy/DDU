from lightning.pytorch.loops import _PredictionLoop, _TrainingEpochLoop
from .EvaluationLoop import customEvaluationLoop
from lightning.pytorch.trainer.states import RunningStage, TrainerFn

class customTrainingEpochLoop(_TrainingEpochLoop):
    def __init__(self, trainer, min_steps = None, max_steps = -1):
        super().__init__(trainer, min_steps = min_steps, max_steps = max_steps)

        self.val_loop = customEvaluationLoop(
            trainer, TrainerFn.FITTING, RunningStage.VALIDATING, verbose=False, inference_mode=False
        )
