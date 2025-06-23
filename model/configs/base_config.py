class BaseConfig:
    def __init__(self):
        self.presenting_dims = [784, 512, 64]  # 特征降维过程
        self.embedding_dims = [64, 16, 2]      # 潜在特征压缩过程

        self.epochs = 1000
        self.batch_size = 6000
        self.lr = 1e-3

        self.warmup_epochs = 150  # 预热阶段轮数
        self.main_epochs = 700    # 主训练阶段轮数
        self.stable_epochs = 150  # 稳定阶段轮数

        self.k = 15
        self.temparature = 0.15

    def load_config(self, config: dict):
        self.presenting_dims = config["presenting_dims"]
        self.embedding_dims = config["embedding_dims"]

        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]

        self.warmup_epochs = config["warmup_epochs"]
        self.main_epochs = config["main_epochs"]
        self.stable_epochs = config["stable_epochs"]

    def get_input_dim(self):
        # 如果 presenting_dims 不为空，返回其第一个元素
        if self.presenting_dims:
            return self.presenting_dims[0]
        # 如果 presenting_dims 为空，尝试从 embedding_dims 获取
        elif self.embedding_dims:
            return self.embedding_dims[0]
        # 如果两个列表都为空，抛出异常
        else:
            raise ValueError(
                "Both presenting_dims and embedding_dims are empty")

    def get_output_dim(self):
        # 如果 embedding_dims 不为空，返回其最后一个元素
        if self.embedding_dims:
            return self.embedding_dims[-1]
        # 如果 embedding_dims 为空，尝试从 presenting_dims 获取
        elif self.presenting_dims:
            return self.presenting_dims[-1]
        # 如果两个列表都为空，抛出异常
        else:
            raise ValueError(
                "Both presenting_dims and embedding_dims are empty")

    def get_latent_dim(self):
        return self.presenting_dims[-1]

    def to_dict(self):
        return {
            "presenting_dims": self.presenting_dims,
            "embedding_dims": self.embedding_dims,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "warmup_epochs": self.warmup_epochs,
            "main_epochs": self.main_epochs,
            "stable_epochs": self.stable_epochs
        }
