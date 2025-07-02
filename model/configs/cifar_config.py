from .base_config import BaseConfig


class CifarConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.presenting_dims = [768, 512, 128, 64]
        self.embedding_dims = [64, 32, 8, 2]

        self.epochs = 1000
        self.batch_size = 1000
        self.lr = 1e-2

        self.k = 10
        self.temparature = 0.2

        self.warmup_epochs = 150
        self.main_epochs = 700
        self.stable_epochs = 150
