from ...predict import Predict
from .lexicon import Lexicon


class CRNNPredict(Predict):

    def __init__(self, model, lexicon: Lexicon, transform=None):
        super().__init__(model, transform)
        self._lexicon = lexicon

    def process(self, img):
        x = self._process(img)[0]
        idx_maxs = x.argmax(dim=1)
        words = []
        for idx in idx_maxs:
            word = self._lexicon[idx]
            words.append(word)

        return words
