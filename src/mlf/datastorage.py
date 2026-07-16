import re
from mlc.datasets import Dataset

class DataStorage:
    def __init__(self, dataset: Dataset, cache_path: str = ".cache"):
        self.ds = dataset
        self.cleaned_texts: list[str] = []
        self.dialogues: list[str] = []

    def load_dialogues(self, max_dialogues: int = -1):
        raw_texts = [
            self.ds.get_text(i)
            for i in range(self.ds.max_texts())
        ]

        dialogues = []
        for text in raw_texts:
            if text is None or not text.strip():
                continue

            turns = re.split(r'#Person\d+#:', text)
            turns = [t.strip() for t in turns if t.strip()]

            formatted_turns = []
            for i, turn in enumerate(turns):
                role_token = "<user>" if i % 2 == 0 else "<ai>"
                formatted_turns.append(f"{role_token} {turn}")
                self.cleaned_texts.append(turn)

            if formatted_turns:
                dialog = "\n".join(formatted_turns) + " <eos>"
                dialogues.append(dialog)

        self.dialogues = dialogues[:max_dialogues] if max_dialogues > 0 else dialogues
