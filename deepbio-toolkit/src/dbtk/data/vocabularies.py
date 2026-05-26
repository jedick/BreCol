from itertools import product

def dna(kmer: int = 1):
    return list(map(str, map("".join, product("ACGT", repeat=kmer))))

class Vocabulary:
    def __init__(self, tokens):
        self._token_to_id = {}
        self._id_to_token = []
        self.update(["[PAD]", "[UNK]", *tokens])

    def add(self, token):
        if token in self._token_to_id:
            return
        self._token_to_id[token] = len(self._id_to_token)
        self._id_to_token.append(token)

    def update(self, tokens):
        for token in tokens:
            self.add(token)

    def __call__(self, tokens):
        return map(self.__getitem__, tokens)

    def __getitem__(self, key):
        return self._token_to_id.get(key, self._token_to_id["[UNK]"])

    def __len__(self):
        return len(self._id_to_token)


class DnaVocabulary(Vocabulary):
    def __init__(self, kmer: int = 1):
        super().__init__(dna(kmer))