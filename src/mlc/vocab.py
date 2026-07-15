import os
import re

from collections import Counter

def get_vocab(
    texts: list[str],
    alphabet: str,
    top_n: int = 3000,
    cache_path: str | None = None
) -> list[str] | None:
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return f.read().split("\n")

    counter = Counter()

    counter["<pad>"] = 999999
    counter["<unk>"] = 999998
    for char in alphabet:
        counter[char] += 999997

    for text in texts:
        text = text.lower()
        words = re.findall(r"[a-z]+|[^\sa-z]", text)

        for i, word in enumerate(words):

            if len(word) <= 4:
                counter[word] += 1
            else:
                for n in [3, 4]:
                    for j in range(len(word) - n + 1):
                        counter[word[j:j+n]] += 1

    top_items = [item[0] for item in counter.most_common(top_n)]
    return top_items
