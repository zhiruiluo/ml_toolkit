from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
class TQDMProgressBarToLogger(TQDMProgressBar):
    def __init__(self, logger, refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate, process_position)
        
    def init_sanity_tqdm(self) -> Tqdm:
        return super().init_sanity_tqdm()