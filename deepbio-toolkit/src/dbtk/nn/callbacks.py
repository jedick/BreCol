from lightning.pytorch.callbacks import Callback
from typing import Any, Dict

class ConfigCheckpoint(Callback):
    """Callback to save config in checkpoint."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['config'] = self.config

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if 'config' in checkpoint:
            self.config = checkpoint['config']
