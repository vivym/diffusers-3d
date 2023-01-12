from pytorch_lightning.callbacks import RichProgressBar as _RichProgressBar


class RichProgressBar(_RichProgressBar):
    def __init__(self):
        super().__init__(leave=True)
