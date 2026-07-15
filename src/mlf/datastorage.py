from mlc.datasets import Dataset

class DataStorage:
    def __init__(
        self,
        dataset: Dataset,
        cache_path: str = ".cache",
    ):
        self.ds = dataset
        self.cleaned_texts: list[str] = []
        self.pairs: list[tuple[str, str]] = []

    def load_pairs(self, max_pairs: int = -1):
        raw_texts = [
            self.ds.get_text(i).lower()
                for i in range(self.ds.max_texts())
        ]

        pairs = []
        for text in raw_texts:
            if text is None:
                continue

            pair = []
            for replic in text.split("\n"):
                cleared = " ".join(replic.split()[1:])
                if len(pair) == 2:
                    pairs.append(pair.copy())
                    pair = []

                pair.append(cleared)
                self.cleaned_texts.append(cleared)

        self.pairs = pairs[:max_pairs]
