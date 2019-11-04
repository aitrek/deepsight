"""Class for label lexicon."""
import os
import pickle


class Lexicon:

    def __init__(self, dump_dir: str):
        self._dump_dir = dump_dir
        self._data = [""]
        self._index = {"": 0}

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int):
        return self._data[index]

    def index(self, char: str):
        # get character index in self._data
        # 0 will be returned if the character not exists in the lexicon
        return self._index.get(char, 0)

    def add(self, chars: str):
        for c in chars:
            if c not in self._index:
                self._index[c] = len(self._data)
                self._data.append(c)

    def dump(self):
        if not os.path.exists(self._dump_dir):
            os.makedirs(self._dump_dir)
        dump_name = os.path.join(self._dump_dir, "lexicon.pkl")
        with open(dump_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, dump_path: str):
        with open(dump_path, "rb") as f:
            return pickle.load(f)
