from functools import lru_cache
from typing import Dict

import numpy as np
from sklearn.decomposition import PCA

from functools2 import flip
from pkl import pkl


class Word2Vec:
    def __init__(self,
                 word2vec: Dict[str, np.array],
                 word2idx: Dict[str, int],
                 d: int = 2,
                 random_state: int = 42):
        self._word2vec = word2vec
        self._vecs = np.array(list(word2vec.values()))
        self._word2idx = word2idx
        self.d = d
        self.random_state = random_state

    @lru_cache()
    def _pca(self):
        pca = PCA(n_components=self.d, random_state=self.random_state)
        pca.fit(self._vecs)
        return pca

    @lru_cache()
    @pkl('_word2vec_pca.npy', dump=flip(np.save), load=np.load)
    def _word2vec_pca(self):
        return self._pca().transform(self._vecs)

    def __getattr__(self, name: str) -> np.array:
        return self._word2vec_pca()[self._word2idx[name]]

    def __getitem__(self, key) -> np.array:
        if isinstance(key, str):
            return getattr(self, key)
        return self._pca().transform(key.reshape(-1, 1)).reshape(-1)
