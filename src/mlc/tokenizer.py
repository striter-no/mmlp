import re

class TokenEngine:
    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.base = len(self.vocab)

        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.pad_id = self.token_to_id.get("<pad>", 0)
        self.unk_id = self.token_to_id.get("<unk>", 1)

        regex_order = sorted(self.vocab, key=len, reverse=True)
        pattern = "|".join(re.escape(t) for t in regex_order if t not in ["<pad>", "<unk>"])
        self.tokenizer_re = re.compile(pattern)

    def tokenize(self, text: str) -> list[int]:
        text = text.lower()
        tokens = self.tokenizer_re.findall(text)

        ids = []
        for t in tokens:
            if t in self.token_to_id:
                ids.append(self.token_to_id[t])
            else:
                for char in t:
                    ids.append(self.token_to_id.get(char, self.unk_id))
        return ids

    def tokenize_to_str(self, text: str) -> list[str]:
        text = text.lower()
        tokens = self.tokenizer_re.findall(text)

        out = []
        for t in tokens:
            if t in self.token_to_id:
                out.append(t)
            else:
                for char in t:
                    out.append(char if char in self.token_to_id else "<unk>")
        return out

    def detokenize(self, ids: list[int]) -> str:
        out = ""
        for i in ids:
            if 0 <= i < len(self.vocab):
                token = self.vocab[i]
                if token not in ["<pad>", "<unk>"]:
                    out += token

        out = re.sub(r'\s+', ' ', out).strip()
        return out
