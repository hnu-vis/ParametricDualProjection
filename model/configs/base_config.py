class BaseConfig:
    def __init__(self):
        self.presenting_dims = [784, 512, 64]
        self.embedding_dims = [64, 16, 2]
        self.epochs = 1000
        self.batch_size = 6000
        self.lr = 1e-3
        self.warmup_epochs = 150
        self.main_epochs = 700
        self.stable_epochs = 150

        self.k = 10
        self.temparature = 0.20

        self.embedding_weight = 1.0
        self.auxiliary_weight = 1.0
        self.latent_weight = 1.0
        self.reconstruction_weight = 1.0

    def load_config(self, config: dict):
        """Load configuration from a dictionary, dynamically setting attributes."""
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: '{key}' is not a valid attribute of {self.__class__.__name__}")

    def to_dict(self):
        """Convert class attributes to a dictionary, excluding methods and private attributes."""
        return {
            key: value
            for key, value in vars(self).items()
            if not key.startswith('_') and not callable(value)
        }

    def get_input_dim(self):
        if self.presenting_dims:
            return self.presenting_dims[0]
        elif self.embedding_dims:
            return self.embedding_dims[0]
        else:
            raise ValueError("Both presenting_dims and embedding_dims are empty")

    def get_output_dim(self):
        if self.embedding_dims:
            return self.embedding_dims[-1]
        elif self.presenting_dims:
            return self.presenting_dims[-1]
        else:
            raise ValueError("Both presenting_dims and embedding_dims are empty")

    def get_latent_dim(self):
        return self.presenting_dims[-1]