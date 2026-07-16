import torch
import os

from mlc.network import AutoregressionNetwork, text_to_ids
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

        self._model = AutoregressionNetwork(
            context_len=settings.context_len,
            dropout=settings.dropout,
            embed_dim=settings.embedding_dims,
            hidden=settings.hidden_neurons,
            pad_id=tokenizer.engine.pad_id,
            eos_id=tokenizer.engine.eos_id,
            num_layers=settings.encoder_layers,
            use_attention=True,
            vocab_size=tokenizer.engine.base
        ).to(self.device)

        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"[network] using {torch.cuda.device_count()} GPUs")
            self._model = torch.nn.DataParallel(self._model)

    @property
    def raw_model(self):
        if isinstance(self._model, torch.nn.DataParallel):
            return self._model.module
        return self._model

    def save_to_file(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.raw_model.state_dict(), path)

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

        network.raw_model.load_state_dict(
            torch.load(model_path, map_location=network.device, weights_only=False)
        )
        network.raw_model.to(network.device)
        network.raw_model.eval()

        return network, settings

    def predict(self, input_text, temperature=None, top_k=None) -> str:
        temp = self.settings.default_temp if temperature is None else temperature
        topk = self.settings.default_top_k if top_k is None else top_k

        self._model.eval()
        q_ids = torch.tensor(
            text_to_ids(
                self.tokenizer.engine,
                self.tokenizer.filter_text(input_text),
                self.raw_model.context_len
            ),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            generated_ids = self.raw_model(q_ids, temperature=temp, top_k=topk)

        ids_list = generated_ids.squeeze(0).tolist()

        return self.tokenizer.engine.detokenize(ids_list)

    def predict_stream(self, input_text, temperature=None, top_k=None):
        temp = self.settings.default_temp if temperature is None else temperature
        topk = self.settings.default_top_k if top_k is None else top_k

        self._model.eval()
        q_ids = torch.tensor(
            text_to_ids(
                self.tokenizer.engine,
                self.tokenizer.filter_text(input_text),
                self.raw_model.context_len
            ),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)

        B = q_ids.size(0)

        last_text = ""
        with torch.no_grad():
            q_emb = self.raw_model.embed(q_ids)
            enc_out, h = self.raw_model.encoder(q_emb)
            h = h[-1]
            mask = self.raw_model._make_mask(q_ids) if self.raw_model.use_attention else None

            prev_id = torch.full((B,), self.raw_model.bos_id, dtype=torch.long, device=self.device)
            generated_ids = []
            finished = torch.zeros(B, dtype=torch.bool, device=self.device)

            for t in range(self.raw_model.context_len):
                prev_emb = self.raw_model.embed(prev_id)
                h = self.raw_model.cell(prev_emb, h)

                if self.raw_model.use_attention:
                    context, _ = self.raw_model.attention(h, enc_out, enc_out, mask)
                    logits = self.raw_model.head(torch.cat([h, context], dim=-1))
                else:
                    logits = self.raw_model.head(h)

                if t > 0:
                    logits[torch.arange(B, device=logits.device), prev_id] -= 10.0

                if topk > 0:
                    top_logits, top_indices = torch.topk(logits, topk, dim=-1)
                    probs = torch.softmax(top_logits / temp, dim=-1)
                    idx = torch.multinomial(probs, 1).squeeze(-1)
                    prev_id = top_indices.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
                else:
                    prev_id = logits.argmax(-1)

                prev_id = torch.where(finished, torch.full_like(prev_id, self.raw_model.pad_id), prev_id)
                finished = finished | (prev_id == self.raw_model.eos_id)

                curr_id = prev_id.item()
                generated_ids.append(curr_id)

                current_text = self.tokenizer.engine.detokenize(generated_ids)
                yield current_text[len(last_text):]

                last_text = current_text
                if finished.all():
                    break
