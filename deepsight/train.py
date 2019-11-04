"""Class to train model"""
import toml
from torch.utils.tensorboard import SummaryWriter


class Train:

    def __init__(self, log_dir):
        self._writer = SummaryWriter(log_dir)

    def run(self, net, criterion, train_loader, test_loader):
        raise NotImplementedError


class DynamicTrain(Train):
    """
    Args:
        config_path (str): Path of the train configuration file which is
            in format of toml. The
    configs file:
    """

    def __init__(self, config_path: str, log_dir: str):
        super().__init__(log_dir)
        self._config_path = config_path
        self._configs = self._load_configs()

    def _load_configs(self):
        return toml.load(self._config_path)

    def update(self):
        self._configs = self._load_configs()

    def run(self, net, criterion, train_loader, test_loader):
        pass





