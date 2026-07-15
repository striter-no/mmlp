from hashlib import sha256
from typing import Counter

import datasets
from datasets import DownloadMode, load_dataset
from datasets.load import load_from_disk

import os
import random

class Dataset:
    def __init__(
        self,
        datasetsFolder = "./.datasets",
        column_text: str | None = "text",
        column_title: str | None = "title"
    ) -> None:
        self.dataset: datasets.Dataset | None = None
        self.folder = datasetsFolder
        self._vocab_cached: list | None = None

        self._column_text = column_text
        self._column_title = column_title

    def is_loaded(self):
        return self.dataset is not None

    def load(self, path: str, name: str, force_reload: bool = False):
        cache_name = sha256((path + name).encode()).hexdigest()
        save_path = os.path.join(self.folder, cache_name)
        if not os.path.exists(save_path) or force_reload:
            self.dataset = load_dataset(
                path, name,
                split="train",
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
            )

            os.makedirs(self.folder, exist_ok=True)
            self.dataset.save_to_disk(save_path)
        else:
            self.dataset = load_from_disk(save_path)

    def get_text(self, inx: int) -> str | None:
        if not self.is_loaded() or inx >= self.max_texts() or self._column_text is None:
            return None
        return self.dataset[inx][self._column_text]

    def get_title(self, inx: int) -> str | None:
        if not self.is_loaded() or inx >= self.max_texts() or self._column_title is None:
            return None
        return self.dataset[inx][self._column_title]

    def get_filtered(self, inx: int, allowed_alpha: str, max_len: int) -> str | None:
        if not self.is_loaded() or inx >= self.max_texts() or self._column_text is None:
            return None

        raw = self.dataset[inx][self._column_text].lower()
        cleared = "".join([c for c in raw if c in allowed_alpha])
        cleared = cleared.replace('ё', 'е')

        return " ".join(cleared[:max_len].split())

    def max_texts(self) -> int:
        if not self.is_loaded():
            return 0
        return len(self.dataset)


if __name__ == "__main__":
    ds = Dataset()
    ds.load("wikimedia/wikipedia", "20231101.en", force_reload=True)

    alphabet_en = " etaoinshrdlcumwfgypbvkjxqz,."
    alphabet_ru = " йцукенгшщзхъфывапролджэячсмитьбю,."

    random_inx = random.randint(0, ds.max_texts() - 1)

    title = ds.get_title(random_inx)
    text = ds.get_filtered(random_inx, alphabet_en, 80 * 40)

    print(f"title: {title}\n")
    print(f"text:\n{text}")
