import torch
import os
from attr import asdict

from mlc.network import AutoregrssionNetwork, text_to_ids
from mlf.tokenizer import Tokenizer
from mlf.settings import ModelSettings
from mlc.datasets import Dataset
from mlf.datastorage import DataStorage

class Network:
    def __init__(
        self,
        tokenizer: Tokenizer,
        settings: ModelSettings,
        device: str = "auto"
    ) -> None:
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = tokenizer
        self.settings = settings

        self._model = AutoregrssionNetwork(
            context_len=settings.context_len,
            dropout=settings.dropout,
            embed_dim=settings.embedding_dims,
            hidden=settings.hidden_neurons,
            pad_id=tokenizer.engine.pad_id,
            use_attention=True,
            vocab_size=tokenizer.engine.base
        ).to(self.device)

    def save_to_file(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self._model.state_dict(), path)

    @staticmethod
    def load_from_file(model_path: str, settings_path: str, device: str = "auto"):
        settings = ModelSettings.load_from_file(settings_path)

        dial_ds = Dataset(column_text=settings.column_text, column_title=settings.column_title)
        dial_ds.load(settings.dataset_path, settings.dataset_name)

        storage = DataStorage(cache_path=".cache", dataset=dial_ds)
        storage.load_pairs(settings.pairs_num)

        tokenizer = Tokenizer(
            alphabet=settings.alphabet,
            cache_path=".cache",
            storage=storage
        )
        tokenizer.load_tokens(settings.tokens_num)

        network = Network(tokenizer=tokenizer, settings=settings, device=device)

        print(f"[network] Loading weights from {model_path}")
        network._model.load_state_dict(torch.load(model_path, map_location=network.device))
        network._model.to(network.device)
        network._model.eval()

        return network, settings

    def predict(self, input_text, temperature=None, top_k=None) -> str:
        temp = self.settings.default_temp if temperature is None else temperature
        topk = self.settings.default_top_k if top_k is None else top_k

        self._model.eval()
        q_ids = torch.tensor(
            text_to_ids(
                self.tokenizer.engine,
                self.tokenizer.filter_text(input_text),
                self._model.context_len
            ),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            generated_ids = self._model(q_ids, temperature=temp, top_k=topk)

        return self.tokenizer.engine.detokenize(
            generated_ids.squeeze(0).tolist()
        )
