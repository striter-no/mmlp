import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

import logging
logger = logging.getLogger(__name__)

class TokenEngine:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

        self.pad_id  = self.tokenizer.token_to_id("<pad>")
        self.bos_id  = self.tokenizer.token_to_id("<bos>")
        self.eos_id  = self.tokenizer.token_to_id("<eos>")
        self.unk_id  = self.tokenizer.token_to_id("<unk>")
        self.user_id = self.tokenizer.token_to_id("<user>")
        self.ai_id   = self.tokenizer.token_to_id("<ai>")

        self.base = self.tokenizer.get_vocab_size()

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 8000, save_path: str | None = None) -> "TokenEngine":
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>", "<user>", "<ai>"],
            initial_alphabet=ByteLevel.alphabet()
        )

        tokenizer.train_from_iterator(texts, trainer)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            tokenizer.save(save_path)
            logger.info(f"[tokenizer] saved to {save_path}")

        return cls(tokenizer)

    @classmethod
    def load(cls, path: str) -> "TokenEngine":
        tokenizer = Tokenizer.from_file(path)
        return cls(tokenizer)

    def tokenize(self, text: str) -> list[int]:
        enc = self.tokenizer.encode(text)
        return enc.ids

    def tokenize_to_str(self, text: str) -> list[str]:
        enc = self.tokenizer.encode(text)
        return enc.tokens

    def detokenize(self, ids: list[int]) -> str:
        # Фильтруем все управляющие / рольные токены
        skip = {self.pad_id, self.bos_id, self.eos_id, self.unk_id, self.user_id, self.ai_id}
        clean_ids = [i for i in ids if i not in skip]
        return self.tokenizer.decode(clean_ids)
