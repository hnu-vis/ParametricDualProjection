from .base_config import BaseConfig


class GeneSampleConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.presenting_dims = [2081, 1024, 512, 256]
        self.embedding_dims = [256, 128, 32, 8, 2]

        self.epochs = 1000
        self.batch_size = 80
        self.lr = 1e-3

        self.warmup_epochs = 150
        self.main_epochs = 700
        self.stable_epochs = 150


class GeneFeatureConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.presenting_dims = [714, 512, 128, 64]
        self.embedding_dims = [64, 32, 8, 2]

        self.epochs = 1000
        self.batch_size =210
        self.lr = 1e-3

        self.warmup_epochs = 150
        self.main_epochs = 700
        self.stable_epochs = 150
