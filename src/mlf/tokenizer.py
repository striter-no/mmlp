from mlc import vocab
from mlc.tokenizer import TokenEngine

from .datastorage import DataStorage

class Tokenizer:
    def __init__(self, alphabet: str, storage: DataStorage, cache_path: str = ".vocab.txt"):
        self.storage = storage
        self.cache_path = cache_path

        self.alphabet = alphabet
        self.engine = TokenEngine([])
        self.tokens = []

    def filter_text(self, text: str) -> str:
        return "".join([c for c in text if c in self.alphabet])

    def load_tokens(self, max_tokens: int):
        ds_vocab = vocab.get_vocab(
            self.storage.cleaned_texts,
            alphabet=self.alphabet,
            top_n=max_tokens
        )

        if ds_vocab is None:
            raise RuntimeError("Failed to load vocab")

        self.tokens = ds_vocab
        self.engine = TokenEngine(self.tokens)

    def tokenize(self, text: str) -> list[int]:
        text = self.filter_text(text)
        tokens = self.engine.tokenize(text)

        return tokens

    def tokenize_str(self, text: str) -> list[str]:
        text = self.filter_text(text)
        tokens = self.engine.tokenize_to_str(text)

        return tokens
