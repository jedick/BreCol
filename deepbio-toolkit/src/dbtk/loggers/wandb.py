from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger as WandbLoggerBase
from lightning.pytorch.loggers.utilities import _scan_checkpoints
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from typing_extensions import override

from .._utils import export

@export
class WandbLogger(WandbLoggerBase):
    @property
    @override
    def save_dir(self):
        """
        This corrects the implementation of `save_dir` in `WandbLogger` to
        return the directory of the experiment instead of the directory of the
        logger.
        """
        if rank_zero_only.rank > 0:
            return None
        return self.experiment.dir

    @override
    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # log iteratively all new checkpoints
        for timestamp, path, score, tag in checkpoints:
            self.experiment.save(path, base_path=self.save_dir, policy="end")
            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[path] = timestamp
