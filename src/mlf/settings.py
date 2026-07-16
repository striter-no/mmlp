import json
import os

from attr import dataclass
from attr import asdict

@dataclass
class ModelSettings:
    dataset_path: str
    dataset_name: str
    column_text: str
    column_title: str
    pairs_num: int
    tokens_num: int
    context_len: int
    embedding_dims: int
    hidden_neurons: int
    default_temp: float
    default_top_k: int
    alphabet: str
    encoder_layers: int = 2
    dropout: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ModelSettings":
        return ModelSettings(**d)

    def save_to_file(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def load_from_file(path: str) -> "ModelSettings":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ModelSettings.from_dict(data)
