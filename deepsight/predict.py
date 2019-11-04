"""Base class for predict"""


class Predict:
    """
    Args:
        model: Concrete model.
        transform: A function/transform that takes in a sample and
            returns a transformed version.
    """

    def __init__(self, model, transform=None):
        self._model = model
        self._transform = transform

    def _process(self, x):
        if self._transform:
            x = self._transform(x).unsqueeze(0)
        return self._model(x)

    def process(self, x):
        # x = self._process(x)
        # ...
        raise NotImplementedError
