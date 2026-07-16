import os
from mlc.tokenizer import TokenEngine
from .datastorage import DataStorage

import logging
logger = logging.getLogger(__name__)

class Tokenizer:
    def __init__(self, storage: DataStorage, cache_path: str = ".cache"):
        self.storage = storage
        self.cache_path = os.path.join(cache_path, "tokenizer.json")
        self.engine = None

    def filter_text(self, text: str) -> str:
        return text.replace('\r', '').strip()

    def load_tokens(self, max_tokens: int):
        if os.path.exists(self.cache_path):
            logger.info(f"[tokenizer] loading BPE tokenizer from cache: {self.cache_path}")
            self.engine = TokenEngine.load(self.cache_path)
        else:
            if not self.storage.cleaned_texts:
                raise RuntimeError("No texts found in storage to train tokenizer.")

            self.engine = TokenEngine.train(
                texts=self.storage.cleaned_texts,
                vocab_size=max_tokens,
                save_path=self.cache_path
            )

        logger.info(f"[tokenizer] vocab size: {self.engine.base}")

    def tokenize(self, text: str) -> list[int]:
        text = self.filter_text(text)
        return self.engine.tokenize(text)

    def tokenize_str(self, text: str) -> list[str]:
        text = self.filter_text(text)
        return self.engine.tokenize_to_str(text)
